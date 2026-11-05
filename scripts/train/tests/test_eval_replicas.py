import copy
import json
import tempfile
import unittest
from pathlib import Path
from scripts.train.shared.eval_replicas import load_replica_plan, replica_for_rank


def manifest(world=8):
    tp = world // 8
    return dict(schema_version=1, training_world_size=world, request_batch_size=2,
                max_num_seqs=4, replicas=[dict(replica_id=i, node_rank=i*tp//8,
                devices=list(range(i*tp%8, i*tp%8+tp)), url=f'http://node{i*tp//8}:{8000+i}',
                group_port=51216+i, sender_rank=i*tp//8*8+(i*tp%8+tp)%8) for i in range(8)])


class ReplicaPlanTests(unittest.TestCase):
    def load(self, data, world=None):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'replicas.json'
            path.write_text(json.dumps(data))
            return load_replica_plan(path, world_size=world)

    def test_routing(self):
        for world in (8, 16):
            plan = self.load(manifest(world), world)
            self.assertEqual([replica_for_rank(plan, r)['replica_id'] for r in range(world)],
                             [r//(world//8) for r in range(world)])

    def test_reject_invalid(self):
        cases = []
        for field in ('url', 'group_port', 'devices'):
            data = manifest()
            data['replicas'][1][field] = copy.deepcopy(data['replicas'][0][field])
            cases.append(data)
        data = manifest(); data['replicas'][0]['sender_rank'] = 0; cases.append(data)
        data = manifest(16); data['replicas'][0]['sender_rank'] = 8; cases.append(data)
        data = manifest(); data['request_batch_size'] = True; cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.load(data)
        with self.assertRaises(ValueError):
            self.load(manifest(), 16)


class ReplicaRuntimeTests(unittest.TestCase):
    def test_sender_proxy_forwards_real_device(self):
        from types import SimpleNamespace
        from scripts.train.shared.vllm import SenderAccelerator
        actual = SimpleNamespace(process_index=3, is_main_process=False, device='cuda:3')
        proxy = SenderAccelerator(actual, 3)
        self.assertTrue(proxy.is_main_process)
        self.assertEqual(proxy.device, 'cuda:3')
        self.assertFalse(SenderAccelerator(actual, 0).is_main_process)

    def test_all_clients_cleanup_after_failure(self):
        from types import SimpleNamespace
        from scripts.train.shared.eval_lifecycle import SharedEvaluationServer
        events = []
        clients = [SimpleNamespace(base_url=f'http://{i}', headers={}, timeout=1) for i in range(3)]
        server = SharedEvaluationServer(clients, SimpleNamespace(process_index=0, num_processes=1, is_main_process=True), release_memory=lambda: None)
        def request(method, path, **kwargs):
            events.append((server.client.base_url, path))
            if path == '/openapi.json':
                return {'paths': {p: {v: {}} for p, v in [('/sleep','post'),('/wake_up','post'),('/is_sleeping','get')]}}
            if path == '/is_sleeping': return {'is_sleeping': True}
        server._request = request
        server.initialize()
        events.clear()
        with self.assertRaises(RuntimeError):
            with server.generation(lambda: (_ for _ in ()).throw(ValueError('fail'))):
                pass
        self.assertEqual([url for url,path in events if path == '/sleep'], ['http://0','http://1','http://2'])

class OrderedSyncTests(unittest.TestCase):
    def test_each_rank_visits_every_replica_in_order(self):
        import torch  # Load before patch.dict restores sys.modules.
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.train.shared.vllm import SFTWeightSync
        events = []
        class Generation:
            def __init__(self, **options):
                self.accelerator = options['accelerator']
                self.url = options['server_base_url']
                self.model = options['model']
                self.vllm_client = SimpleNamespace(base_url=self.url, session=SimpleNamespace(headers={}))
            def sync_weights(self):
                events.append((self.url, self.accelerator.is_main_process))
        fake = SimpleNamespace(VLLMGeneration=Generation)
        exporter = SimpleNamespace(install_policy_weight_export=lambda generation: None)
        plan = manifest(16)
        for rank in range(16):
            accelerator = SimpleNamespace(process_index=rank, num_processes=16, unwrap_model=lambda model:model)
            trainer = SimpleNamespace(accelerator=accelerator, model_wrapped=object(), processing_class=None)
            with patch('scripts.train.shared.eval_replicas.replica_plan_from_environment', return_value=plan), patch.dict(sys.modules, {'trl.generation.vllm_generation':fake, 'scripts.train.rl.rollout.weights':exporter}):
                sync = SFTWeightSync(trainer, 'http://legacy')
                events.clear(); sync.sync_weights()
                self.assertEqual([url for url, sender in events], [r['url'] for r in plan['replicas']])
                self.assertEqual([sender for url, sender in events], [r['sender_rank']==rank for r in plan['replicas']])

class StartupCleanupTests(unittest.TestCase):
    def test_api_failure_still_sleeps_every_replica(self):
        from types import SimpleNamespace
        from scripts.train.shared.eval_lifecycle import SharedEvaluationServer
        clients = [SimpleNamespace(base_url=str(i)) for i in range(3)]
        lifecycle = SharedEvaluationServer(clients, SimpleNamespace(is_main_process=True, num_processes=1, process_index=0))
        events = []
        def check():
            if lifecycle.client is clients[0]:
                raise ValueError('invalid API')
        lifecycle._check_api = check
        lifecycle._sleep = lambda: events.append(lifecycle.client.base_url)
        with self.assertRaises(RuntimeError):
            lifecycle.initialize()
        self.assertEqual(events, ['0', '1', '2'])
        self.assertFalse(lifecycle.initialized)



def sync_failure_worker(rank, directory):
    import sys
    import torch.distributed as dist
    from datetime import timedelta
    from types import SimpleNamespace
    from unittest.mock import patch
    from scripts.train.shared.vllm import SFTWeightSync
    from scripts.train.shared.eval_lifecycle import SharedEvaluationServer
    dist.init_process_group('gloo', init_method='file://' + directory + '/store',
                            rank=rank, world_size=2, timeout=timedelta(seconds=3))
    visited, slept = [], []
    class Generation:
        def __init__(self, **options):
            self.url = options['server_base_url']
            self.accelerator = options['accelerator']
            self.vllm_client = SimpleNamespace(base_url=self.url, session=SimpleNamespace(headers={}))
        def sync_weights(self):
            visited.append(self.url)
            dist.barrier()  # Full tensor transport finished on all ranks.
            if rank == 1 and len(visited) == 1:
                raise ValueError('private reset-prefix failure')
    accelerator = SimpleNamespace(process_index=rank, num_processes=2,
                                  unwrap_model=lambda model:model)
    trainer = SimpleNamespace(accelerator=accelerator, model_wrapped=object(), processing_class=None)
    clients = [SimpleNamespace(base_url=str(i)) for i in range(8)]
    lifecycle = SharedEvaluationServer(clients, SimpleNamespace(process_index=rank, num_processes=2, is_main_process=rank==0), release_memory=lambda: None)
    lifecycle.initialized = True
    lifecycle._request = lambda *args, **kwargs: {'is_sleeping': False}
    lifecycle._sleep = lambda: slept.append(lifecycle.client.base_url)
    error = None
    try:
        with patch('scripts.train.shared.eval_replicas.replica_plan_from_environment', return_value=manifest()), patch.dict(sys.modules, {
            'trl.generation.vllm_generation': SimpleNamespace(VLLMGeneration=Generation),
            'scripts.train.rl.rollout.weights': SimpleNamespace(install_policy_weight_export=lambda generation:None)}):
            with lifecycle.generation(SFTWeightSync(trainer, 'legacy').sync_weights):
                raise AssertionError('must not generate')
    except RuntimeError as caught:
        error = str(caught)
    Path(directory, str(rank)+'.json').write_text(json.dumps(dict(visited=visited, slept=slept, error=error)))
    dist.destroy_process_group()


class DistributedSyncBoundaryTests(unittest.TestCase):
    def test_sender_post_transfer_failure_stops_all_ranks_and_cleans_all(self):
        import multiprocessing
        with tempfile.TemporaryDirectory() as directory:
            ctx = multiprocessing.get_context('spawn')
            processes = [ctx.Process(target=sync_failure_worker, args=(rank,directory)) for rank in range(2)]
            try:
                for process in processes: process.start()
                for process in processes:
                    process.join(20)
                    self.assertEqual(process.exitcode, 0)
                for rank in range(2):
                    record = json.loads(Path(directory,str(rank)+'.json').read_text())
                    self.assertEqual(record['visited'], [manifest()['replicas'][0]['url']])
                    self.assertEqual(record['slept'], [str(i) for i in range(8)] if rank == 0 else [])
                    self.assertIn('weight synchronization', record['error'])
            finally:
                for process in processes:
                    if process.is_alive(): process.terminate()
                    if process.pid is not None: process.join()

class BoundedResetTests(unittest.TestCase):
    def test_reset_uses_explicit_timeout_without_retry_session(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.train.shared.vllm import install_bounded_prefix_reset
        generation = SimpleNamespace(accelerator=SimpleNamespace(is_main_process=True), vllm_client=SimpleNamespace(base_url='http://replica', session=SimpleNamespace(headers={'X-Test':'yes'})))
        with patch('requests.post') as post:
            post.return_value.status_code = 200
            install_bounded_prefix_reset(generation)
            generation.vllm_client.reset_prefix_cache()
            post.assert_called_once_with('http://replica/reset_prefix_cache', headers={'X-Test':'yes'}, timeout=(5, 60))
            post.return_value.json.assert_called_once()

if __name__ == '__main__':
    unittest.main()
