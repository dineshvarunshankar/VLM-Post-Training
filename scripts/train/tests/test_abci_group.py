import copy
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.train.launch_abci_group import validate_plan, stop_process, wait_runs


def fixture():
    return {'nodes': {'a': [
        {'config':'scripts/train/configs/runs/a.yaml','train_gpu':0,'eval_gpu':1,'http_port':8010,'group_port':51220},
        {'config':'scripts/train/configs/runs/b.yaml','train_gpu':2,'eval_gpu':3,'http_port':8011,'group_port':51221}]}}


class GroupTests(unittest.TestCase):
    def test_cot_paths_and_repository_boundary(self):
        plan=fixture()
        plan['nodes']['a'][0]['config']='scripts/train/configs/cot/qwen3.5-9b.yaml'
        self.assertEqual(len(validate_plan(plan)['a']),2)
        for value in ('/tmp/run.yaml','scripts/train/configs/cot/../runs/a.yaml',
                      'scripts/train/configs/other/a.yaml'):
            plan['nodes']['a'][0]['config']=value
            with self.subTest(path=value),self.assertRaises(ValueError): validate_plan(plan)

    def test_independent_pairs(self):
        self.assertEqual(len(validate_plan(fixture())['a']),2)

    def test_cot_plan_matches_model_ranks_and_ports(self):
        from urllib.parse import urlsplit
        from scripts.train.shared.cot_inspect import resolve_inspection_run
        root=Path(__file__).resolve().parents[3]
        nodes=validate_plan(json.loads((root/'scripts/train/configs/abci/cot_two_nodes.json').read_text()))
        self.assertEqual(len(nodes),2)
        for entries in nodes.values():
            entry=entries[0]
            run=resolve_inspection_run(root/entry['config'],'/fixture/media','/fixture/splits')['run']
            self.assertEqual(run['answer_format'],'cot')
            self.assertEqual(run['execution']['training_gpus'],len(entry['train_gpu']))
            self.assertEqual(urlsplit(run['execution']['vllm_server_url']).port,entry['http_port'])

    def test_multi_gpu_group_and_environment(self):
        from scripts.train.launch_abci_group import environment
        plan = fixture()
        plan['nodes']['a'] = [dict(plan['nodes']['a'][0], train_gpu=[0,1,2,3], eval_gpu=4)]
        entry = validate_plan(plan)['a'][0]
        self.assertEqual(environment(entry)['ABCI_TRAIN_GPU'], '0,1,2,3')
        plan['nodes']['a'][0]['train_gpu'] = [0,0]
        with self.assertRaises(ValueError): validate_plan(plan)

    def test_reject_resource_collisions(self):
        for field,value in [('train_gpu',1),('eval_gpu',0),('http_port',8010),('group_port',51220),('group_port',8010)]:
            plan=fixture(); plan['nodes']['a'][1][field]=value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_reject_duplicate_config_across_nodes(self):
        plan=fixture(); plan['nodes']['b']=copy.deepcopy(plan['nodes']['a'])
        with self.assertRaises(ValueError): validate_plan(plan)

    def test_reject_out_of_range_gpu_and_port(self):
        for field,value in [('train_gpu',8),('eval_gpu',-1),('http_port',65536),('group_port',80),('train_gpu',True)]:
            plan=fixture();plan['nodes']['a'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_plan(plan)

    def test_stop_only_owned_process_group(self):
        proc=Mock(pid=123);proc.poll.return_value=None
        with patch('os.killpg') as kill:
            stop_process(proc)
            kill.assert_called_once_with(123,signal.SIGTERM)
            proc.wait.assert_called_once()

    def test_one_failed_run_does_not_cancel_sibling(self):
        bad=Mock();bad.poll.return_value=1
        good=Mock();good.poll.side_effect=[None,0]
        with tempfile.TemporaryDirectory() as d, patch('time.sleep'):
            result=wait_runs({'bad':bad,'good':good},Path(d))
            self.assertEqual(result,{'bad':1,'good':0})
            good.terminate.assert_not_called()
            self.assertEqual(json.loads((Path(d)/'results.json').read_text()),result)

    def test_checked_in_plan_matches_run_ports_and_preserves_batch(self):
        from urllib.parse import urlsplit
        from scripts.train.shared.run_config import load_run, compile_run
        root=Path(__file__).resolve().parents[3]
        plan=validate_plan(json.loads((root/'scripts/train/configs/abci/sft_two_nodes.json').read_text()))
        self.assertEqual([len(x) for x in plan.values()],[1]*4)
        seen=[]
        for entries in plan.values():
            for entry in entries:
                run=load_run(root/entry['config']);compiled=compile_run(run)
                self.assertEqual(urlsplit(run['execution']['vllm_server_url']).port,entry['http_port'])
                self.assertEqual(run['execution']['training_gpus'], len(entry['train_gpu']) if isinstance(entry['train_gpu'], list) else 1)
                self.assertEqual(run['sft']['effective_batch_size'],16)
                self.assertEqual(run['sft']['training']['num_train_epochs'],8 if run['model'].startswith('qwen') else 5)
                seen.append(run['run']['name'])
        self.assertEqual(len(set(seen)),4)
        self.assertNotIn('qwen9b-stabilized-medical',seen)

    def test_worker_success_and_failure_clean_up_servers(self):
        import socket
        import subprocess
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from scripts.train.launch_abci_group import worker
        def free_port():
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0))
                return sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            server=root/'fake_server.py'
            server.write_text('from http.server import BaseHTTPRequestHandler, HTTPServer\nimport os\nclass Handler(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200); self.end_headers()\nHTTPServer(("127.0.0.1",int(os.environ["ABCI_SERVER_PORT"])),Handler).serve_forever()\n')
            launcher=root/'launch.sh'
            launcher.write_text(f'#!/bin/bash\nif [[ "$1" == server ]]; then exec {sys.executable} {server}; fi\nif [[ "$2" == bad ]]; then exit 9; fi\nsleep 1\nexit 0\n')
            good=dict(fixture()['nodes']['a'][0],config='good',http_port=free_port())
            bad=dict(fixture()['nodes']['a'][1],config='bad',http_port=free_port())
            with patch('scripts.train.launch_abci_group.LAUNCHER',launcher), ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(worker,e,root/e['config']) for e in (bad,good)]
                self.assertEqual([f.result(timeout=20) for f in futures],[9,0])
            for entry in (good,bad):
                with socket.socket() as sock:
                    self.assertNotEqual(sock.connect_ex(('127.0.0.1',entry['http_port'])),0)
                self.assertEqual((root/entry['config']/'exit-code').read_text().strip(),'0' if entry['config']=='good' else '9')
