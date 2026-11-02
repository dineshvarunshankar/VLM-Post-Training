"""CPU HTTP tests for shared training/evaluation GPU ownership."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from scripts.train.shared import eval_lifecycle


class SharedEvalTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.sleeping = False
        self.fail_path = None
        self.paths = {'/sleep': {'post': {}}, '/wake_up': {'post': {}}, '/is_sleeping': {'get': {}}}
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200); self.end_headers()
                payload = {'paths': outer.paths} if self.path == '/openapi.json' else {'is_sleeping': outer.sleeping}
                self.wfile.write(json.dumps(payload).encode())
            def do_POST(self):
                outer.events.append(self.path)
                self.send_response(500 if outer.fail_path == self.path else 200)
                self.end_headers()  # vLLM control endpoints return an EMPTY success body.
                if outer.fail_path != self.path:
                    if self.path.startswith('/sleep?'):
                        outer.sleeping = True
                    elif self.path == '/wake_up?tags=kv_cache':
                        outer.sleeping = False
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = SimpleNamespace(base_url=f'http://127.0.0.1:{self.server.server_port}', headers={}, timeout=2)
        self.state = SimpleNamespace(is_main_process=True, num_processes=1, process_index=0)
        self.lifecycle = eval_lifecycle.SharedEvaluationServer(self.client, self.state,
            release_memory=lambda: self.events.append('release'))
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
    def test_full_cycle_releases_memory_and_sleeps_after_generation(self):
        self.lifecycle.initialize()
        with self.lifecycle.generation(lambda: self.events.append('sync')):
            self.events.append('generate')
            self.assertFalse(self.sleeping)
        self.assertEqual(self.events, ['/sleep?level=1', 'release', '/wake_up?tags=weights',
            'sync', 'release', '/wake_up?tags=kv_cache', 'generate', '/sleep?level=1'])
        self.assertTrue(self.sleeping)
    def test_generation_failure_returns_server_to_sleep_and_preserves_error(self):
        self.lifecycle.initialize()
        with self.assertRaisesRegex(RuntimeError, 'generation body'):
            with self.lifecycle.generation(lambda: None):
                raise ValueError('generation failed')
        self.assertTrue(self.sleeping)
    def test_sync_failure_does_not_wake_kv_and_returns_server_to_sleep(self):
        self.lifecycle.initialize()
        def fail():
            raise ValueError('sync failed')
        with self.assertRaisesRegex(RuntimeError, 'weight synchronization'):
            with self.lifecycle.generation(fail):
                self.fail('generation must not run')
        self.assertNotIn('/wake_up?tags=kv_cache', self.events)
        self.assertTrue(self.sleeping)
    def test_body_and_cleanup_failure_preserves_body_error(self):
        self.lifecycle.initialize()
        with self.assertRaisesRegex(RuntimeError, 'generation body') as raised:
            with self.lifecycle.generation(lambda: None):
                self.fail_path = '/sleep?level=1'
                raise ValueError('original error')
        self.assertTrue(any('Also failed' in note for note in raised.exception.__notes__))

    def test_failed_wake_is_cleaned_up_without_starting_generation(self):
        self.lifecycle.initialize()
        self.fail_path = '/wake_up?tags=weights'
        with self.assertRaisesRegex(RuntimeError, 'wake weights'):
            with self.lifecycle.generation(lambda: self.fail('sync must not run')):
                self.fail('generation must not run')
        self.assertEqual(self.events[-1], '/sleep?level=1')
    def test_missing_sleep_api_fails_before_training(self):
        self.paths.pop('/sleep')
        with self.assertRaisesRegex(RuntimeError, 'sleep API'):
            self.lifecycle.initialize()
        self.assertEqual(self.events, [])
    def test_uninitialized_lifecycle_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'initialized'):
            with self.lifecycle.generation(lambda: None):
                self.fail('generation must not run')
    def test_second_generation_cycle_syncs_again(self):
        self.lifecycle.initialize()
        for _ in range(2):
            with self.lifecycle.generation(lambda: self.events.append('sync')):
                pass
        self.assertEqual(self.events.count('sync'), 2)
        self.assertEqual(self.events.count('/sleep?level=1'), 3)
    def test_sleep_failure_does_not_allow_training_to_resume(self):
        self.lifecycle.initialize()
        with self.assertRaisesRegex(RuntimeError, 'sleep after evaluation'):
            with self.lifecycle.generation(lambda: None):
                self.fail_path = '/sleep?level=1'

if __name__ == '__main__':
    unittest.main()
