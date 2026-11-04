"""Bounded API retries, a process-wide cooldown, and safe request metrics."""
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
import math
import os
import random
import re
import sys
from threading import Lock
import time


_metrics = ContextVar('cot_request_metrics', default=None)


class InvalidResponse(ValueError):
    """A model response failed validation; source-input errors use ValueError."""


class GatewayUnavailable(RuntimeError):
    """Stop the batch without marking all remaining clips as failed."""


def gateway_health(api_base, model, api_key, *, checks=5, interval=5):
    """Check the configured model listing without uploading media or generating tokens."""
    import httpx
    results = []
    if not api_key:
        return {'healthy': False, 'reason': 'GEMINI_API_KEY is missing', 'checks': results}
    if not api_base:
        return {'healthy': True, 'reason': 'Direct Gemini route; no proxy health endpoint', 'checks': results}
    model = model.removeprefix('litellm_proxy/')
    for index in range(checks):
        started = time.monotonic()
        try:
            # Fresh connections also exercise different load-balancer routes.
            with httpx.Client(timeout=10) as client:
                response = client.get(api_base.rstrip('/') + '/models',
                                      headers={'Authorization': 'Bearer ' + api_key})
            item = {'status': response.status_code, 'server': response.headers.get('server'),
                    'seconds': round(time.monotonic() - started, 3)}
            results.append(item)
            if response.status_code != 200:
                return {'healthy': False, 'reason': f'Gateway HTTP {response.status_code}', 'checks': results}
            data = response.json()
            models = data.get('data') if isinstance(data, dict) else None
            if not isinstance(models, list) or not any(isinstance(row, dict) and row.get('id') == model for row in models):
                return {'healthy': False, 'reason': 'Configured model missing from gateway listing', 'checks': results}
        except (httpx.HTTPError, ValueError) as error:
            return {'healthy': False, 'reason': f'Gateway health check failed: {type(error).__name__}', 'checks': results}
        if index + 1 < checks:
            time.sleep(interval)
    return {'healthy': True, 'reason': f'{checks} consecutive model-list checks passed', 'checks': results}


def reset_metrics():
    _metrics.set(None)


def last_metrics():
    return dict(_metrics.get() or {})


def safe_error_details(error):
    """Retain API diagnostics without logging credentials or inline media."""
    message = str(error)
    key = os.getenv('GEMINI_API_KEY')
    if key:
        message = message.replace(key, '[REDACTED]')
    message = re.sub(r'(?i)bearer\s+[^\s\"\'\\]+', 'Bearer [REDACTED]', message)
    message = re.sub(r'data:[^\s\"\']+', '[MEDIA REDACTED]', message)
    message = re.sub(r'[A-Za-z0-9+/=_-]{100,}', '[LONG VALUE REDACTED]', message)
    return {'type': type(error).__name__, 'status_code': getattr(error, 'status_code', None),
            'message': message[:2000]}


def retry_after(headers):
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def response_metrics(response):
    usage = response.usage.model_dump(mode='json') if response.usage else {}
    hidden = getattr(response, '_hidden_params', {}) or {}
    headers = hidden.get('additional_headers') or {}
    cost = hidden.get('response_cost')
    if cost is None:
        cost = next((v for k, v in headers.items() if k.lower() == 'x-litellm-response-cost'), None)
    try:
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0:
            cost = None
    except (TypeError, ValueError):
        cost = None
    return {
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens'),
        'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
        'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
        'reported_cost_usd': cost,
    }


class RequestRunner:
    def __init__(self):
        self._lock = Lock()
        self._gateway_lock = Lock()
        self._ready_at = 0.0
        self._gateway_failures = 0
        self._gateway_needs_check = False
        self._gateway_blocked = None

    def raise_if_unavailable(self):
        with self._lock:
            reason = self._gateway_blocked
        if reason:
            raise GatewayUnavailable(reason)

    def _check_gateway(self, payload):
        self.raise_if_unavailable()
        with self._gateway_lock:
            self.raise_if_unavailable()
            with self._lock:
                needed = self._gateway_needs_check
            if not needed:
                return
            health = gateway_health(payload.get('api_base'),
                                    payload.get('model', ''), os.getenv('GEMINI_API_KEY'))
            with self._lock:
                if health['healthy']:
                    self._gateway_failures = 0
                    self._gateway_needs_check = False
                else:
                    self._gateway_blocked = health['reason']
            self.raise_if_unavailable()
            print('Gateway health checks passed; resuming requests.', file=sys.stderr, flush=True)

    def _gateway_failure(self, status, payload):
        if not payload.get('api_base') or status not in (500, 502, 503, 504):
            return
        with self._lock:
            self._gateway_failures += 1
            opening = self._gateway_failures >= 5 and not self._gateway_needs_check
            if opening:
                self._gateway_needs_check = True
                self._ready_at = max(self._ready_at, time.monotonic() + 60)
        if opening:
            print('Repeated gateway errors: all workers cooling down for 60 seconds, '
                  'then checking gateway health.', file=sys.stderr, flush=True)

    def _defer(self, delay):
        with self._lock:
            self._ready_at = max(self._ready_at, time.monotonic() + delay)

    def _wait(self, deadline):
        while True:
            now = time.monotonic()
            with self._lock:
                delay = max(0.0, self._ready_at - now)
            if now + delay >= deadline:
                raise TimeoutError('API request deadline reached during cooldown')
            if not delay:
                return
            time.sleep(min(delay, 1.0))

    def complete(self, call, payload, *, validate=None, max_retries=2, deadline_seconds=900):
        start = time.monotonic()
        deadline = start + deadline_seconds
        metrics = {'attempts': 0, 'retries': 0, 'rate_limit_retries': 0,
                   'validation_retries': 0, 'validation_errors': [],
                   'wait_seconds': 0.0, 'elapsed_seconds': 0.0}
        _metrics.set(metrics)
        feedback = None
        responses = 0
        try:
            for attempt in range(max_retries + 1):
                self.raise_if_unavailable()
                waiting = time.monotonic()
                self._wait(deadline)
                self._check_gateway(payload)
                metrics['wait_seconds'] += time.monotonic() - waiting
                if time.monotonic() >= deadline:
                    raise TimeoutError('API request deadline reached during gateway checks')
                kwargs = {**payload, 'num_retries': 0,
                          'timeout': min(payload.get('timeout', 300), deadline - time.monotonic())}
                if feedback:
                    kwargs['messages'] = [*payload['messages'], {'role': 'user', 'content': feedback}]
                metrics['attempts'] += 1
                metrics['retries'] = attempt
                try:
                    response = call(**kwargs)
                except Exception as error:
                    import litellm
                    status = getattr(error, 'status_code', None)
                    transient = status in (408, 429, 500, 502, 503, 504) or isinstance(
                        error, (litellm.Timeout, litellm.APIConnectionError))
                    if not transient:
                        raise
                    self._gateway_failure(status, payload)
                    headers = getattr(getattr(error, 'response', None), 'headers', {}) or {}
                    delay = retry_after(headers)
                    if delay is None:
                        delay = min(60, 2 ** (attempt + 1)) + random.uniform(0, 1)
                    # Even an exhausted request must notify the other workers.
                    self._defer(delay)
                    if attempt == max_retries:
                        raise
                    if status == 429:
                        metrics['rate_limit_retries'] += 1
                    continue
                with self._lock:
                    if not self._gateway_needs_check:
                        self._gateway_failures = 0
                # Include usage from rejected responses; those calls also incur cost.
                for key, value in response_metrics(response).items():
                    previous = metrics.get(key)
                    metrics[key] = (value if not responses else
                                    previous + value if previous is not None and value is not None else None)
                responses += 1
                if validate is not None:
                    try:
                        validate(response)
                    except InvalidResponse as error:
                        reason = str(error)[:500]
                        metrics['validation_errors'].append(reason)
                        if attempt == max_retries:
                            raise
                        metrics['validation_retries'] += 1
                        feedback = (
                            f'The previous response failed validation: {reason}. '
                            'Return a complete corrected JSON response using the required schema. '
                            'Use only supplied evidence values and well-formed evidence tags. '
                            'Preserve the supplied human answer exactly.')
                        continue
                return response
        finally:
            metrics['elapsed_seconds'] = round(time.monotonic() - start, 3)
            metrics['wait_seconds'] = round(metrics['wait_seconds'], 3)


REQUESTS = RequestRunner()
