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
                timeout=(10, self.timeout), **kwargs,
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

    def generate(self, packets, *, policy_version, temperature, top_p, top_k,
                 repetition_penalty, max_tokens, schemas=None, presence_penalty=0.0, min_p=0.0, seed=None):
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
                parsed.append((batch_index, packet['token_ids'], completion, logps))
            return parsed

        ordered = [None] * len(packets)
        with ThreadPoolExecutor(max_workers=min(self.max_concurrent_requests, len(groups))) as pool:
            for responses in pool.map(send, groups.values()):
                for index, prompt, completion, logps in responses:
                    ordered[index] = (prompt, completion, logps)
        return {
            'prompt_ids': [row[0] for row in ordered],
            'completion_ids': [row[1] for row in ordered],
            'logprobs': [row[2] for row in ordered],
        }
