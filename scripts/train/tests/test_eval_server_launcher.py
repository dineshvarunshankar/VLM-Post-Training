"""Behavioral tests for all-GPU evaluation launch construction and supervision."""
import os
from pathlib import Path
import sys
import json
import socket
import subprocess
import signal
import time
import ctypes
import tempfile
import unittest

from scripts.train.psc import eval_servers as servers


class ReplicaLauncherTests(unittest.TestCase):
    def test_all_devices_and_local_alternate_senders(self):
        for hosts, tp in ((['one'], 1), (['one', 'two'], 2)):
            plan = servers.replica_manifest(hosts)
            self.assertEqual(plan['training_world_size'], len(hosts) * 8)
            self.assertEqual((plan['request_batch_size'], plan['max_num_seqs']), (2, 4))
            self.assertEqual(len(plan['replicas']), 8)
            covered = set()
            for replica in plan['replicas']:
                self.assertEqual(len(replica['devices']), tp)
                covered.update((replica['node_rank'], gpu) for gpu in replica['devices'])
                self.assertEqual(replica['sender_rank'] // 8, replica['node_rank'])
                self.assertNotIn(replica['sender_rank'] % 8, replica['devices'])
            self.assertEqual(len(covered), len(hosts) * 8)

    def test_command_preserves_profile_and_isolates_replica(self):
        replica = servers.replica_manifest(['one', 'two'])['replicas'][5]
        command, env = servers.server_command(replica, weights='/weights', model='gemma',
            max_length=8192, utilization=.63, limits='{"video": 1}', kwargs='{"num_frames": 100}',
            extra_arguments=['--reasoning-parser', 'gemma'], cache_root=Path('/cache'),
            max_num_seqs=4)
        for flag, value in (('--tensor-parallel-size', '2'), ('--max-num-seqs', '4'),
                            ('--max-model-len', '8192'), ('--mm-processor-kwargs', '{"num_frames": 100}'),
                            ('--gpu-memory-utilization', '0.63'), ('--port', '8001')):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3')
        self.assertTrue(env['VLLM_CACHE_ROOT'].startswith('/cache/replica-5/'))
        self.assertEqual(env['TRIAGE_VLLM_GROUP_PORT'], str(replica['group_port']))

    def test_failed_child_cleans_up_other_process_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = servers.ServerSupervisor(Path(directory))
            survivor = supervisor.start(0, [sys.executable, '-c', 'import time; time.sleep(60)'], os.environ.copy())
            failure = supervisor.start(1, [sys.executable, '-c', 'raise SystemExit(7)'], os.environ.copy())
            failure.wait(timeout=5)
            with self.assertRaisesRegex(RuntimeError, 'replica 1'):
                with supervisor:
                    supervisor.wait_until(lambda: False, timeout=2, interval=.01)
            self.assertIsNotNone(survivor.poll())
            self.assertIsNotNone(failure.poll())

    def test_wait_is_bounded_and_cleanup_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                with servers.ServerSupervisor(Path(directory)) as supervisor:
                    child = supervisor.start(0, [sys.executable, '-c', 'import time; time.sleep(60)'], os.environ.copy())
                    supervisor.wait_until(lambda: False, timeout=.03, interval=.01)
            self.assertIsNotNone(child.poll())

    def test_supervisor_cli_starts_all_servers_and_cleans_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = servers.replica_manifest(['127.0.0.1'])
            reservations = []
            for replica in manifest['replicas']:
                reservation = socket.socket()
                reservation.bind(('127.0.0.1', 0))
                reservations.append(reservation)
                replica['url'] = f'http://127.0.0.1:{reservation.getsockname()[1]}'
            fake = root / 'vllm'
            fake.write_text(f'#!{sys.executable}\n' +
                'import sys\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n'
                'class Handler(BaseHTTPRequestHandler):\n'
                ' def do_GET(self):\n  self.send_response(200); self.end_headers()\n'
                'HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port")+1])), Handler).serve_forever()\n')
            fake.chmod(0o700)
            servers.atomic_json(root / 'manifest.json', manifest)
            servers.atomic_json(root / 'plan.json', {'run': {'model': 'test'}, 'experiment': {'max_length': 123}})
            servers.atomic_json(root / 'profile.json', dict(weights='/unused', utilization=.63,
                limits='{}', kwargs='{}', extra_arguments=[]))
            for reservation in reservations:
                reservation.close()
            marker = root / 'training-started'
            result = subprocess.run([sys.executable, '-m', 'scripts.train.psc.eval_servers',
                '--manifest', str(root / 'manifest.json'), '--node-rank', '0',
                '--plan', str(root / 'plan.json'), '--profile', str(root / 'profile.json'),
                '--job-root', str(root), '--cache-root', str(root / 'cache'), '--timeout', '10', '--',
                sys.executable, '-c', 'from pathlib import Path; import sys; Path(sys.argv[1]).touch()', str(marker)],
                env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH']),
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(marker.exists())
            pids = list((root / 'eval-node-0').glob('*.pid'))
            self.assertEqual(len(pids), 8)
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid.read_text()), 0)

    def test_training_cleanup_kills_descendants_after_parent_exit(self):
        # Adopt grandchildren so the test can reap them and prove the group is gone.
        libc = ctypes.CDLL(None)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        try:
            for parent_exits in (True, False):
                with self.subTest(parent_exits=parent_exits), tempfile.TemporaryDirectory() as directory:
                    pidfile = Path(directory) / 'descendant.pid'
                    descendant_code = ('import os,signal,time; from pathlib import Path; '
                        'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                        f'Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)')
                    parent_code = ('import subprocess,sys,time; from pathlib import Path; '
                        f'subprocess.Popen([sys.executable,"-c",{descendant_code!r}]); '
                        f'p=Path({str(pidfile)!r}); '
                        '\nwhile not p.exists(): time.sleep(.01)\n' +
                        ('raise SystemExit(0)' if parent_exits else 'time.sleep(60)'))
                    parent = subprocess.Popen([sys.executable, '-c', parent_code], start_new_session=True)
                    descendant_pid = None
                    try:
                        deadline = time.monotonic() + 5
                        while not pidfile.exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue(pidfile.exists())
                        descendant_pid = int(pidfile.read_text())
                        if parent_exits:
                            parent.wait(timeout=5)
                        servers.cleanup_training_group(parent, grace=.05)
                        deadline = time.monotonic() + 5
                        status = (0, 0)
                        while time.monotonic() < deadline:
                            status = os.waitpid(descendant_pid, os.WNOHANG)
                            if status[0]:
                                break
                            time.sleep(.01)
                        self.assertEqual(status[0], descendant_pid)
                        self.assertEqual(os.waitstatus_to_exitcode(status[1]), -signal.SIGKILL)
                        descendant_pid = None
                        with self.assertRaises(ProcessLookupError):
                            os.killpg(parent.pid, 0)
                    finally:
                        try:
                            os.killpg(parent.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        parent.wait()
                        if descendant_pid is not None:
                            os.waitpid(descendant_pid, 0)
        finally:
            libc.prctl(36, previous.value, 0, 0, 0)

    def test_runtime_manifest_matches_shared_validator(self):
        from scripts.train.shared.eval_replicas import load_replica_plan
        with tempfile.TemporaryDirectory() as directory:
            for hosts in (['one'], ['one', 'two']):
                path = Path(directory) / 'replicas.json'
                servers.atomic_json(path, servers.replica_manifest(hosts))
                plan = load_replica_plan(path, world_size=len(hosts) * 8)
                self.assertEqual(len(plan['replicas']), 8)

    def test_allocation_records_actual_phase_gpu_counts(self):
        from scripts.train.psc.launch import allocation_metadata
        for nodes in (1, 2):
            metadata = allocation_metadata(nodes, [], replica_file='/runtime/replicas.json')
            self.assertEqual(metadata['evaluation_gpus'], nodes * 8)
            self.assertEqual(metadata['training_gpus'], nodes * 8)
            self.assertEqual(metadata['evaluation_replicas'], 8)
            self.assertEqual(metadata['evaluation_tensor_parallel_size'], nodes)
            legacy = allocation_metadata(nodes, [], replica_file=None)
            self.assertEqual(legacy['evaluation_gpus'], nodes)
            self.assertEqual(legacy['evaluation_mode'], 'legacy')

    def test_manifest_resume_preserves_saved_endpoint(self):
        from scripts.train.psc.launch import configure_server_endpoint
        run = {'execution': {'vllm_server_url': 'http://original:8000'}}
        configure_server_endpoint(run, 'replacement', replica_file='/runtime/manifest.json')
        self.assertEqual(run['execution']['vllm_server_url'], 'http://original:8000')
        configure_server_endpoint(run, 'replacement', replica_file=None)
        self.assertEqual(run['execution']['vllm_server_url'], 'http://replacement:8000')
