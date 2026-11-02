"""Real two-process Gloo checks for rank-consistent handoff failures."""
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.train.tests import test_shared_eval_lifecycle as lifecycle_tests


def run_rank(rank, rendezvous, url, failure, results):
    import torch.distributed as distributed
    from scripts.train.shared.eval_lifecycle import SharedEvaluationServer
    distributed.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    client = SimpleNamespace(base_url=url, headers={}, timeout=5)
    state = SimpleNamespace(is_main_process=rank == 0, num_processes=2, process_index=rank)
    lifecycle = SharedEvaluationServer(client, state, release_memory=lambda: None)
    reached_body = False
    try:
        lifecycle.initialize()
        def sync():
            if failure == 'sync' and rank == 1:
                raise ValueError('rank one sync failure')
        with lifecycle.generation(sync):
            reached_body = True
            if failure == 'body' and rank == 1:
                raise ValueError('rank one body failure')
        results.put((rank, 'resumed', reached_body))
    except Exception as error:
        results.put((rank, str(error), reached_body))
    finally:
        distributed.destroy_process_group()


class SharedEvaluationDistributedTests(unittest.TestCase):
    setUp = lifecycle_tests.SharedEvalTests.setUp
    tearDown = lifecycle_tests.SharedEvalTests.tearDown

    def check_failure(self, failure):
        context = multiprocessing.get_context('spawn')
        results = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [context.Process(target=run_rank,
                args=(rank, str(Path(directory) / 'rendezvous'), self.client.base_url, failure, results))
                for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(40)
                    self.assertFalse(process.is_alive(), 'Collective lifecycle hung after rank failure')
                    self.assertEqual(process.exitcode, 0)
                values = sorted(results.get(timeout=2) for _ in processes)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate(); process.join(5)
                results.close()
        self.assertTrue(self.sleeping)
        self.assertEqual(self.events.count('/sleep?level=1'), 2)
        for rank, message, reached_body in values:
            self.assertIn('failed on rank(s) 1', message)
            self.assertEqual(reached_body, failure == 'body')
        if failure == 'sync':
            self.assertNotIn('/wake_up?tags=kv_cache', self.events)

    def test_sync_failure_stops_both_ranks_and_sleeps(self):
        self.check_failure('sync')

    def test_generation_failure_stops_both_ranks_and_sleeps(self):
        self.check_failure('body')

if __name__ == '__main__':
    unittest.main()
