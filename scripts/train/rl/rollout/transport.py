"""Validate token-in/token-out vLLM responses before RL loss calculation."""

import math


def parse_rollout_response(response, prompt_ids, *, max_tokens):
    """Read one response from /inference/v1/generate with token echo enabled.

    Token equality checks text alignment only. The caller must independently
    verify that the server received the same processed video features.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('max_tokens must be a positive integer')
    if not isinstance(response, dict):
        raise ValueError('vLLM response must be a JSON object')
    echoed = response.get('prompt_token_ids')
    if (not isinstance(echoed, list) or not echoed
            or any(type(token) is not int or token < 0 for token in echoed)
            or echoed != prompt_ids):
        raise ValueError('vLLM must echo the exact training prompt token IDs')
    choices = response.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('Expected one vLLM completion per request')
    choice = choices[0]
    if choice.get('finish_reason') not in ('stop', 'length'):
        raise ValueError('vLLM completion did not finish normally')
    tokens = choice.get('token_ids')
    if (not isinstance(tokens, list) or not 0 < len(tokens) <= max_tokens
            or any(type(token) is not int or token < 0 for token in tokens)):
        raise ValueError('Invalid or excessive completion token IDs')
    probabilities = choice.get('logprobs')
    entries = probabilities.get('content') if isinstance(probabilities, dict) else None
    if not isinstance(entries, list) or len(entries) != len(tokens):
        raise ValueError('Sampling log probabilities must cover every completion token')
    logprobs = []
    for token, entry in zip(tokens, entries):
        if (not isinstance(entry, dict) or type(entry.get('token_id')) is not int
                or entry['token_id'] != token):
            raise ValueError('Sampling log probability belongs to a different token')
        value = entry.get('logprob')
        if type(value) not in (int, float) or not math.isfinite(value) or value > 0:
            raise ValueError('Sampling log probabilities must be finite and non-positive')
        logprobs.append(float(value))
    return tokens, logprobs
