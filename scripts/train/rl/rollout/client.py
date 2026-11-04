"""HTTP transport for processed native-video RL rollouts."""

import math
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from scripts.train.rl.rollout.transport import parse_rollout_response


class VideoRolloutClient:
    """Use patched vLLM's token/feature endpoint without decoding video again."""

    def __init__(self, base_url, model, *, api_key=None,
                 max_concurrent_requests=8, timeout=600):
        if not isinstance(base_url, str) or not base_url.startswith(('http://', 'https://')):
            raise ValueError('A rollout HTTP(S) URL is required')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('The served model name is required')
        if type(max_concurrent_requests) is not int or max_concurrent_requests < 1:
            raise ValueError('max_concurrent_requests must be positive')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self._checked = False

    def _request(self, method, path, **kwargs):
        try:
            response = requests.request(
                method, self.base_url + path, headers=self.headers,
                # Requests uses the connection socket timeout while sending large video payloads.
                timeout=(self.timeout, self.timeout), **kwargs,
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError) as error:
            # Do not echo request tensors, API credentials, or server response bodies.
            raise RuntimeError(f'Rollout server request failed: {method} {path}') from error
        if not isinstance(result, dict):
            raise ValueError('Rollout server returned a non-object JSON response')
        return result

    def check_compatibility(self):
        if self._request('GET', '/version').get('version') != '0.30.0':
            raise ValueError('Expected the verified patched vLLM 0.30.0 server')
        schema = self._request('GET', '/openapi.json')
        endpoint = schema.get('paths', {}).get('/inference/v1/generate', {})
        definitions = schema.get('components', {}).get('schemas', {})
        mask_supported = any(
            name.startswith('PlaceholderRangeInfo')
            and 'is_embed' in definition.get('properties', {})
            for name, definition in definitions.items()
        )
        if 'post' not in endpoint or not mask_supported:
            raise ValueError('Rollout server lacks the video placeholder-mask interface')
        models = self._request('GET', '/v1/models').get('data', [])
        if not any(item.get('id') == self.model for item in models):
            raise ValueError('Requested model is not advertised by the rollout server')
        self._checked = True

    def check_image_compatibility(self):
        """Require a live server configured to encode at least one native image."""
        try:
            info = self._request('GET', '/server_info', params={'config_format': 'json'})
            settings = info['vllm_config']['model_config']['multimodal_config']
            limits = settings['limit_per_prompt']
            if not isinstance(limits, dict):
                raise TypeError('Invalid multimodal limits')
            # vLLM 0.30.0 MultiModalConfig.get_limit_per_prompt defaults an
            # unspecified modality to 999; explicit limits serialize as dataclasses.
            image = limits.get('image', 999)
            count = image.get('count') if isinstance(image, dict) else image
            ready = (settings.get('language_model_only', False) is False
                     and type(count) is int and count >= 1)
        except (RuntimeError, KeyError, AttributeError, TypeError):
            ready = False
        if not ready:
            raise ValueError('Subject-check evaluation requires native image capacity on the vLLM server. '
                             'Start the future-run server with an image limit of at least one and expose '
                             '/server_info?config_format=json before loading training weights.')

    def check_thinking_budget_compatibility(self, reasoning_parser):
        """Require initialized native reasoning markers; never assume server setup."""
        if reasoning_parser not in ('qwen3','gemma4'):
            raise ValueError('Unsupported CoT reasoning parser')
        if reasoning_parser in getattr(self, '_checked_reasoning_parsers', set()):
            return
        try:
            info = self._request('GET','/server_info',params={'config_format':'json'})
            reasoning = info.get('vllm_config',{}).get('reasoning_config') or {}
            start, end = reasoning.get('_reasoning_start_token_ids'), reasoning.get('_reasoning_end_token_ids')
            native_start, native_end = {'qwen3':('<think>','</think>'),
                                        'gemma4':('<|channel>','<channel|>')}[reasoning_parser]
            ready = (reasoning.get('reasoning_parser')==reasoning_parser
                     and reasoning.get('reasoning_start_str','') in ('',native_start)
                     and reasoning.get('reasoning_end_str','') in ('',native_end)
                     and reasoning.get('_enabled') is True
                     and isinstance(start,list) and bool(start) and isinstance(end,list) and bool(end)
                     and all(type(token) is int and token>=0 for token in start+end))
        except (RuntimeError,AttributeError,TypeError):
            ready = False
        if not ready:
            raise ValueError(f'CoT thinking budget requires an initialized server parser. Start the compatible vLLM server with --reasoning-parser {reasoning_parser} and expose /server_info?config_format=json before training; the existing PSC launcher may require this explicit flag. Remove conflicting --reasoning-config marker overrides.')
        self._checked_reasoning_parsers = getattr(self, '_checked_reasoning_parsers', set()) | {reasoning_parser}

    def check_cot_schema_compatibility(self, reasoning_parser):
        """Require final-only grammar gating and explicit per-request thinking."""
        self.check_thinking_budget_compatibility(reasoning_parser)
        if reasoning_parser in getattr(self, '_checked_cot_schema_parsers', set()):
            return
        try:
            info = self._request('GET', '/server_info', params={'config_format':'json'})
            settings = info.get('vllm_config', {}).get('structured_outputs_config') or {}
            api = self._request('GET', '/openapi.json')
            body = api['paths']['/inference/v1/generate']['post']['requestBody']['content']['application/json']['schema']
            if '$ref' in body:
                body = api['components']['schemas'][body['$ref'].rsplit('/', 1)[-1]]
            field = body.get('properties', {}).get('enable_thinking', {})
            boolean = field.get('type') == 'boolean' or any(
                option.get('type') == 'boolean' for option in field.get('anyOf', []))
            ready = (settings.get('reasoning_parser') == reasoning_parser
                     and settings.get('enable_in_reasoning') is False and boolean)
        except (RuntimeError, KeyError, AttributeError, TypeError):
            ready = False
        if not ready:
            raise ValueError('CoT final-answer schema evaluation requires the vllm-task-thinking.patch, '
                             'a matching --reasoning-parser, and structured_outputs_config.enable_in_reasoning=false. '
                             'Apply the pinned patches in the future-run environment and restart its server.')
        self._checked_cot_schema_parsers = getattr(self, '_checked_cot_schema_parsers', set()) | {reasoning_parser}

    def generate(self, packets, *, policy_version, temperature, top_p, top_k,
                 repetition_penalty, max_tokens, schemas=None, presence_penalty=0.0, min_p=0.0, seed=None,
                 thinking_token_budget=None, reasoning_parser=None, enable_thinking=None):
        """Return one response per entry, grouping shared packets into n-way requests.

        policy_version is set by the weight-synchronization caller. Its cache salt
        prevents prefix-cache reuse across updates; it does not synchronize weights.
        """
        if not packets:
            raise ValueError('A rollout batch cannot be empty')
        if not isinstance(policy_version, str) or not 0 < len(policy_version.strip()) <= 1024:
            raise ValueError('A nonempty policy version of at most 1024 characters is required')
        for name, value in (('temperature', temperature), ('top_p', top_p),
                            ('repetition_penalty', repetition_penalty)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != 'temperature' and value == 0):
                raise ValueError(f'{name} must be positive and finite')
        if type(min_p) not in (int, float) or not math.isfinite(min_p) or not 0 <= min_p <= 1:
            raise ValueError('min_p must be finite and in [0, 1]')
        if top_p > 1 or type(top_k) is not int or top_k < 0:
            raise ValueError('Invalid top_p or top_k')
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        if enable_thinking is not None:
            if type(enable_thinking) is not bool:
                raise ValueError('enable_thinking must be a boolean')
            if not enable_thinking and thinking_token_budget is not None:
                raise ValueError('Direct tasks must not set a thinking token budget')
            self.check_cot_schema_compatibility(reasoning_parser)
        if thinking_token_budget is not None:
            if type(thinking_token_budget) is not int or not 0 <= thinking_token_budget < max_tokens:
                raise ValueError('thinking_token_budget must be a non-negative integer below max_tokens')
            self.check_thinking_budget_compatibility(reasoning_parser)
        if schemas is not None and len(schemas) != len(packets):
            raise ValueError('Each constrained request needs its own answer schema')
        if type(presence_penalty) not in (int, float) or not math.isfinite(presence_penalty):
            raise ValueError('presence_penalty must be finite')
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('seed must be a nonnegative integer')
        groups = {}
        for index, packet in enumerate(packets):
            if not isinstance(packet, dict) or set(packet) != {'token_ids', 'features'}:
                raise ValueError('Rollout packets must contain only token IDs and features')
            key = (id(packet), index if schemas is not None else None)
            groups.setdefault(key, (packet, []))[1].append(index)
        if not self._checked:
            self.check_compatibility()

        def send(group):
            packet, indices = group
            sampling = {
                'n': len(indices), 'temperature': temperature, 'top_p': top_p, 'min_p': min_p,
                'top_k': -1 if top_k == 0 else top_k,
                'repetition_penalty': repetition_penalty,
                'max_tokens': max_tokens, 'logprobs': 0, 'detokenize': False,
                'output_kind': 2,  # FINAL_ONLY preserves every response for n > 1.
            }
            if thinking_token_budget is not None:
                sampling['thinking_token_budget'] = thinking_token_budget
            sampling['presence_penalty'] = presence_penalty
            if schemas is not None:
                sampling['structured_outputs'] = {'json': schemas[indices[0]]}
            if seed is not None:
                sampling['seed'] = seed + indices[0]
            response = self._request('POST', '/inference/v1/generate', json={
                'request_id': f'triage-rl-{uuid.uuid4().hex}',
                'model': self.model, **packet, 'sampling_params': sampling,
                'return_token_ids': True, 'stream': False,
                'cache_salt': policy_version,
                **({'enable_thinking': enable_thinking} if enable_thinking is not None else {}),
            })
            expected = packet['features']['mm_placeholders']
            if response.get('mm_placeholders') != expected:
                raise ValueError('Rollout server changed or discarded visual-token positions')
            choices = response.get('choices')
            if not isinstance(choices, list) or len(choices) != len(indices):
                raise ValueError('Rollout server returned the wrong number of responses')
            indexed = {}
            for choice in choices:
                if not isinstance(choice, dict) or type(choice.get('index')) is not int:
                    raise ValueError('Rollout response lacks an integer choice index')
                index = choice['index']
                if index in indexed or not 0 <= index < len(indices):
                    raise ValueError('Rollout choice indices are duplicated or out of range')
                indexed[index] = choice
            parsed = []
            for choice_index, batch_index in enumerate(indices):
                completion, logps = parse_rollout_response(
                    {**response, 'choices': [indexed[choice_index]]},
                    packet['token_ids'], max_tokens=max_tokens,
                )
                parsed.append((batch_index, packet['token_ids'], completion, logps, indexed[choice_index]['finish_reason']))
            return parsed

        ordered = [None] * len(packets)
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent_requests, len(groups))) as pool:
            for responses in pool.map(send, groups.values()):
                for index, prompt, completion, logps, finish_reason in responses:
                    ordered[index] = (prompt, completion, logps, finish_reason)
        return {
            'prompt_ids': [row[0] for row in ordered],
            'completion_ids': [row[1] for row in ordered],
            'logprobs': [row[2] for row in ordered],
            'finish_reasons': [row[3] for row in ordered],
        }
