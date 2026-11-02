"""Opt-in ownership handoff when vLLM and SFT share physical GPUs.

All ranks must enter each phase. This handles ordinary Python/HTTP failures;
a failed NCCL collective still requires the distributed launcher's timeout.
The server must start with --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1.
"""
from contextlib import contextmanager

import requests


def release_cuda_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class SharedEvaluationServer:
    def __init__(self, client, state, *, release_memory=release_cuda_cache):
        self.client = client
        self.state = state
        self.release_memory = release_memory
        self.initialized = False

    def phase(self, label, operation, *, main_only=False):
        """Complete a phase everywhere before any rank advances or raises."""
        error = None
        result = None
        try:
            if not main_only or self.state.is_main_process:
                result = operation()
        except Exception as caught:
            error = caught
        status = {'rank': self.state.process_index, 'error': type(error).__name__ if error else None}
        if self.state.num_processes > 1:
            import torch.distributed as distributed
            statuses = [None] * self.state.num_processes
            distributed.all_gather_object(statuses, status)
        else:
            statuses = [status]
        failures = [item for item in statuses if item['error']]
        if failures:
            # Do not put remote responses, payloads, or credentials in the message.
            ranks = ', '.join(str(item['rank']) for item in failures)
            raise RuntimeError(f'Shared evaluation {label} failed on rank(s) {ranks}') from error
        return result

    def _request(self, method, path, *, params=None, json_body=False):
        try:
            response = requests.request(method, self.client.base_url + path,
                headers=self.client.headers, timeout=(self.client.timeout, self.client.timeout), params=params)
            response.raise_for_status()
            return response.json() if json_body else None
        except (requests.RequestException, ValueError) as error:
            raise RuntimeError(f'vLLM control request failed: {method} {path}') from error

    def _check_api(self):
        schema = self._request('GET', '/openapi.json', json_body=True)
        paths = schema.get('paths', {})
        if any(verb not in paths.get(path, {}) for path, verb in (
            ('/sleep', 'post'), ('/wake_up', 'post'), ('/is_sleeping', 'get'))):
            raise ValueError('vLLM needs sleep APIs and VLLM_SERVER_DEV_MODE=1')

    def _assert_sleeping(self, expected):
        response = self._request('GET', '/is_sleeping', json_body=True)
        if not isinstance(response, dict) or response.get('is_sleeping') is not expected:
            raise RuntimeError('vLLM did not enter the requested sleep state')

    def _sleep(self):
        self._request('POST', '/sleep', params={'level': 1})
        self._assert_sleeping(True)

    def initialize(self):
        self.phase('sleep API check', self._check_api, main_only=True)
        self.phase('startup sleep', self._sleep, main_only=True)
        self.initialized = True

    @contextmanager
    def generation(self, synchronize_weights):
        if not self.initialized:
            raise RuntimeError('Shared evaluation server must be initialized before loading the trainer model')
        primary_error = None
        try:
            self.phase('release training cache', self.release_memory)
            self.phase('wake weights', lambda: self._request('POST', '/wake_up', params={'tags': 'weights'}), main_only=True)
            self.phase('weight synchronization', synchronize_weights)
            self.phase('release transfer cache', self.release_memory)
            def wake_kv():
                self._request('POST', '/wake_up', params={'tags': 'kv_cache'})
                self._assert_sleeping(False)
            self.phase('wake KV cache', wake_kv, main_only=True)
            body_error = None
            try:
                yield
            except Exception as error:
                body_error = error
            def finish_body():
                if body_error is not None:
                    raise body_error
            self.phase('generation body', finish_body)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                self.phase('sleep after evaluation', self._sleep, main_only=True)
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                # Never turn a failed evaluation or failed cleanup into resumed training.
                primary_error.add_note(f'Also failed to return vLLM to sleep: {cleanup_error}')
