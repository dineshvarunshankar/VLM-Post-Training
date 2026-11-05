"""Runtime-only evaluation topology and bounded per-node server supervision."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def replica_manifest(hosts):
    if len(hosts) not in (1, 2) or len(set(hosts)) != len(hosts):
        raise ValueError('Expected one or two distinct node hosts')
    tp = len(hosts)
    replicas = []
    for replica_id in range(8):
        node, first = divmod(replica_id * tp, 8)
        replicas.append(dict(replica_id=replica_id, node_rank=node,
            devices=list(range(first, first + tp)), url=f'http://{hosts[node]}:{8000 + first // tp}',
            group_port=51216 + replica_id, sender_rank=node * 8 + (tp if first == 0 else 0)))
    return dict(schema_version=1, training_world_size=len(hosts) * 8,
                request_batch_size=1, max_num_seqs=1, replicas=replicas)


def server_command(replica, *, weights, model, max_length, utilization, limits, kwargs,
                   extra_arguments, cache_root, max_num_seqs):
    from urllib.parse import urlsplit
    command = ['vllm', 'serve', str(weights), *extra_arguments,
        '--enable-sleep-mode', '--enforce-eager', '--served-model-name', model,
        '--host', '0.0.0.0', '--port', str(urlsplit(replica['url']).port), '--dtype', 'bfloat16',
        '--tensor-parallel-size', str(len(replica['devices'])), '--max-num-seqs', str(max_num_seqs),
        '--gpu-memory-utilization', str(utilization), '--logprobs-mode', 'processed_logprobs',
        '--max-model-len', str(max_length), '--enable-scale-out', '--enable-mm-embeds',
        '--video-pruning-rate', '0', '--limit-mm-per-prompt', limits,
        '--mm-processor-kwargs', kwargs, '--weight-transfer-config', '{"backend":"nccl"}']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, replica['devices'])), VLLM_SERVER_DEV_MODE='1',
               TRIAGE_VLLM_GROUP_PORT=str(replica['group_port']))
    cache = Path(cache_root) / f"replica-{replica['replica_id']}"
    for key, name in (('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                      ('CUDA_CACHE_PATH', 'cuda'), ('VLLM_CACHE_ROOT', 'vllm')):
        env[key] = str(cache / name)
    return command, env


class ServerSupervisor:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.children = {}
        self.logs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self, replica_id, command, env):
        for key in ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'CUDA_CACHE_PATH', 'VLLM_CACHE_ROOT'):
            if env.get(key):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
        log = (self.root / f'replica-{replica_id}.log').open('ab')
        self.logs.append(log)
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[replica_id] = child
        (self.root / f'replica-{replica_id}.pid').write_text(str(child.pid) + '\n')
        return child

    def check_alive(self):
        for replica_id, child in self.children.items():
            if child.poll() is not None:
                raise RuntimeError(f'Evaluation replica {replica_id} exited with {child.returncode}')

    def wait_until(self, predicate, *, timeout, interval=2):
        deadline = time.monotonic() + timeout
        while True:
            self.check_alive()
            if predicate():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Evaluation server readiness timed out')
            time.sleep(min(interval, max(0, deadline - time.monotonic())))

    def close(self):
        # Always signal the entire group, including descendants of an exited parent.
        for child in self.children.values():
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        for child in self.children.values():
            try:
                child.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for child in self.children.values():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for log in self.logs:
            log.close()


def cleanup_training_group(training, *, grace=10):
    """Stop the owned session even when its launcher has already exited."""
    try:
        os.killpg(training.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        training.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A launcher can exit on TERM while torchrun workers ignore it. Its exit
    # therefore never proves that the owned process group has been cleaned up.
    try:
        os.killpg(training.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    training.wait()


def healthy(url):
    try:
        # Scheduler hostnames must bypass user/system HTTP proxies.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + '/health', timeout=3) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--node-rank', type=int)
    parser.add_argument('--hosts', nargs='+')
    parser.add_argument('--plan')
    parser.add_argument('--profile')
    parser.add_argument('--job-root')
    parser.add_argument('--cache-root')
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('training_command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.hosts:
        atomic_json(args.manifest, replica_manifest(args.hosts))
        return
    manifest = json.loads(Path(args.manifest).read_text())
    plan = json.loads(Path(args.plan).read_text())
    profile = json.loads(Path(args.profile).read_text())
    root = Path(args.job_root)
    failure = root / f'eval-node-{args.node_rank}.failed'
    ready = root / f'eval-node-{args.node_rank}.ready'
    command = args.training_command
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        parser.error('A training command is required')
    def interrupted(signum, frame):
        raise RuntimeError(f'Evaluation supervisor received signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        with ServerSupervisor(root / f'eval-node-{args.node_rank}') as supervisor:
            local = [r for r in manifest['replicas'] if r['node_rank'] == args.node_rank]
            for replica in local:
                server, env = server_command(replica, weights=profile['weights'], model=plan['run']['model'],
                    max_length=plan['experiment']['max_length'], utilization=profile['utilization'],
                    limits=profile['limits'], kwargs=profile['kwargs'], extra_arguments=profile['extra_arguments'],
                    cache_root=args.cache_root, max_num_seqs=manifest['max_num_seqs'])
                supervisor.start(replica['replica_id'], server, env)
            supervisor.wait_until(lambda: all(healthy(r['url']) for r in local), timeout=args.timeout)
            atomic_json(ready, {'replicas': [r['replica_id'] for r in local]})
            nodes = manifest['training_world_size'] // 8
            def all_ready():
                failed = list(root.glob('eval-node-*.failed'))
                if failed:
                    raise RuntimeError(f'Peer evaluation startup failed: {failed}')
                return all((root / f'eval-node-{node}.ready').exists() for node in range(nodes))
            supervisor.wait_until(all_ready, timeout=args.timeout)
            # Runtime preflight sleeps every endpoint before loading training weights.
            training = subprocess.Popen(command, start_new_session=True)
            try:
                while training.poll() is None:
                    supervisor.check_alive()
                    if list(root.glob('eval-node-*.failed')):
                        raise RuntimeError('Peer evaluation supervisor failed')
                    time.sleep(1)
                if training.returncode:
                    raise subprocess.CalledProcessError(training.returncode, command)
            finally:
                cleanup_training_group(training)
    except BaseException as exc:
        atomic_json(failure, {'error': str(exc)})
        raise
    finally:
        ready.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
