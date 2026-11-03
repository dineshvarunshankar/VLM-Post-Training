"""Idle evaluation ranks must not wait in one long collective."""
import importlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from datetime import timedelta


def worker(rank, rendezvous, output, fail):
    import torch.distributed as dist
    from scripts.train.shared.distributed_local_work import run_local_work
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=3))
    calls = []
    def operation():
        calls.append(rank)
        if rank == 0:
            time.sleep(4.5)  # Longer than the process-group timeout.
            if fail:
                raise ValueError('private response must not reach other ranks')
            return ['prediction']
        return []  # No examples on this rank.
    try:
        result = run_local_work('prediction', operation, interval=0.1)
        record = {'result': result, 'calls': calls}
    except RuntimeError as error:
        record = {'error': str(error), 'calls': calls}
    Path(output, str(rank) + '.json').write_text(json.dumps(record))
    dist.destroy_process_group()


class LocalWorkTests(unittest.TestCase):
    def run_distributed(self, fail):
        importlib.import_module('scripts.train.shared.distributed_local_work')
        with tempfile.TemporaryDirectory() as directory:
            ctx = multiprocessing.get_context('spawn')
            processes = [ctx.Process(target=worker, args=(rank, directory + '/store', directory, fail)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(25)
                    self.assertEqual(process.exitcode, 0)
                return [json.loads(Path(directory, str(rank) + '.json').read_text()) for rank in range(2)]
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join()

    def test_slow_prediction_and_empty_rank_complete_once(self):
        rows = self.run_distributed(False)
        self.assertEqual(rows, [{'result': ['prediction'], 'calls': [0]}, {'result': [], 'calls': [1]}])

    def test_failure_reaches_every_rank_without_response_text(self):
        for rank, row in enumerate(self.run_distributed(True)):
            self.assertEqual(row['calls'], [rank])
            self.assertIn('rank(s) 0', row['error'])
            self.assertNotIn('private response', row['error'])

    def test_single_process_preserves_result_and_error(self):
        run = importlib.import_module('scripts.train.shared.distributed_local_work').run_local_work
        self.assertEqual(run('local', lambda: 17), 17)
        with self.assertRaisesRegex(ValueError, 'local failure'):
            run('local', lambda: (_ for _ in ()).throw(ValueError('local failure')))

    def test_explicit_evaluation_timeout(self):
        from scripts.train.shared.config import evaluation_request_timeout
        self.assertEqual(evaluation_request_timeout({}), 600)
        self.assertEqual(evaluation_request_timeout({'evaluation': {'request_timeout_seconds': 1800}}), 1800)
        for invalid in [0, -1, True, float('inf'), float('nan'), '1800']:
            with self.assertRaises(ValueError):
                evaluation_request_timeout({'evaluation': {'request_timeout_seconds': invalid}})


if __name__ == '__main__':
    unittest.main()
