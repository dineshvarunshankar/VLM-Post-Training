#!/usr/bin/env python3
"""Inspect a node allocation, or run its experiments inside an existing PBS job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / 'scripts/train/launch_abci.sh'


def validate_plan(plan):
    if set(plan) != {'nodes'} or not isinstance(plan['nodes'], dict) or not plan['nodes']:
        raise ValueError('Plan must contain a nonempty nodes mapping')
    configs = set()
    for name, entries in plan['nodes'].items():
        if not name.replace('_', '').isalnum() or not isinstance(entries, list) or not 1 <= len(entries) <= 4:
            raise ValueError('Each named node must have one to four experiments')
        gpus, ports = set(), set()
        for entry in entries:
            if set(entry) != {'config','train_gpu','eval_gpu','http_port','group_port'}:
                raise ValueError('Unexpected allocation fields')
            config = entry['config']
            if not isinstance(config, str) or not config.startswith(('scripts/train/configs/runs/', 'scripts/train/configs/cot/')) or '..' in Path(config).parts or not config.endswith('.yaml'):
                raise ValueError('Expected a repository-relative run YAML')
            if config in configs:
                raise ValueError('A run must occur only once in the plan')
            configs.add(config)
            training = entry['train_gpu'] if isinstance(entry['train_gpu'], list) else [entry['train_gpu']]
            if not training:
                raise ValueError('At least one training GPU is required')
            for value in [*training, entry['eval_gpu']]:
                if type(value) is not int or not 0 <= value < 8 or value in gpus:
                    raise ValueError('GPU assignments must be distinct integers from 0 to 7')
                gpus.add(value)
            for field in ('http_port','group_port'):
                value = entry[field]
                if type(value) is not int or not 1024 <= value <= 65535 or value in ports:
                    raise ValueError('HTTP and transfer ports must be distinct unprivileged ports')
                ports.add(value)
    return plan['nodes']


def stop_process(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def wait_runs(processes, directory):
    results = {}
    while len(results) < len(processes):
        for name, process in processes.items():
            if name not in results:
                code = process.poll()
                if code is not None:
                    results[name] = code
                    temp = directory / 'results.json.tmp'
                    temp.write_text(json.dumps(results, indent=2) + '\n')
                    temp.replace(directory / 'results.json')
        if len(results) < len(processes):
            time.sleep(5)
    return results


def environment(entry, *, profile_video_frames=None):
    env = os.environ.copy()
    training = entry['train_gpu'] if isinstance(entry['train_gpu'], list) else [entry['train_gpu']]
    env.update(ABCI_TRAIN_GPU=','.join(map(str, training)), ABCI_EVAL_GPU=str(entry['eval_gpu']),
               ABCI_SERVER_PORT=str(entry['http_port']), TRIAGE_VLLM_GROUP_PORT=str(entry['group_port']),
               ABCI_FFMPEG_BIN=str(REPO.parent.parent / 'tools/ffmpeg/bin'))
    if profile_video_frames is not None:
        if type(profile_video_frames) is not int or profile_video_frames < 2:
            raise ValueError('Measured profile_video_frames must be an integer of at least 2')
        requested = env.get('ABCI_PROFILE_VIDEO_FRAMES')
        if requested is not None:
            if not requested.isascii() or not requested.isdigit() or int(requested) < profile_video_frames:
                raise ValueError('ABCI_PROFILE_VIDEO_FRAMES must be an integer at least as large as '
                                 f'the measured dataset bound ({profile_video_frames})')
        else:
            env['ABCI_PROFILE_VIDEO_FRAMES'] = str(profile_video_frames)
    return env


def interrupted(signum, frame):
    raise InterruptedError(f'Received signal {signum}')


def worker(entry, directory):
    directory.mkdir(parents=True, exist_ok=True)
    server = train = None
    code = 1
    try:
        # Use the same project/container runtime as training, not host Python packages.
        measured = subprocess.check_output(['bash', str(LAUNCHER), 'frame-bound', entry['config']],
                                           cwd=REPO, env=environment(entry), text=True)
        report = json.loads(measured)
        env = environment(entry, profile_video_frames=report['profile_video_frames'])
        report['effective_profile_video_frames'] = int(env['ABCI_PROFILE_VIDEO_FRAMES'])
        (directory/'frame-bound.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Video frame profile: '+json.dumps(report), flush=True)
        with (directory/'server.log').open('w') as log:
            server = subprocess.Popen(['bash',str(LAUNCHER),'server',entry['config']],cwd=REPO,
                                      env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline = time.monotonic() + 1800
        while True:
            if server.poll() is not None:
                raise RuntimeError('Evaluation server exited; see server.log')
            try:
                with urlopen(f"http://127.0.0.1:{entry['http_port']}/health",timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Evaluation server startup exceeded 30 minutes')
            time.sleep(5)
        with (directory/'train.log').open('w') as log:
            train = subprocess.Popen(['bash',str(LAUNCHER),'train',entry['config']],cwd=REPO,
                                     env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while train.poll() is None:
            if server.poll() is not None:
                raise RuntimeError('Evaluation server died during training')
            time.sleep(5)
        code = train.returncode
    finally:
        stop_process(train)
        stop_process(server)
        (directory/'exit-code').write_text(str(code)+'\n')
    return code


def launch_node(entries, node):
    if not os.environ.get('PBS_JOBID') or os.environ.get('RTYPE') != 'rt_HF':
        raise ValueError('Run inside one exclusive rt_HF PBS allocation')
    if not Path(os.environ.get('PBS_LOCALDIR','/nonexistent')).is_dir():
        raise ValueError('PBS_LOCALDIR is required')
    ids = subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True)
    if len(ids.splitlines()) < 8:
        raise ValueError('This plan requires an eight-GPU allocation')
    directory = REPO.parent / 'scratch/sft-groups' / os.environ['PBS_JOBID'] / node
    directory.mkdir(parents=True, exist_ok=False)
    locks, processes = [], {}
    try:
        lock_dir = REPO.parent.parent / 'runtime/vlm-post-training/locks'
        lock_dir.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            name = Path(entry['config']).stem
            lock = (lock_dir / (name+'.lock')).open('a')
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for port in (entry['http_port'],entry['group_port']):
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1',port))
            with (directory/(name+'.preflight.log')).open('w') as log:
                subprocess.run(['bash',str(LAUNCHER),'check',entry['config']],cwd=REPO,env=environment(entry),
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        (directory/'allocation.json').write_text(json.dumps(entries,indent=2)+'\n')
        (directory/'code-commit').write_text(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True))
        for entry in entries:
            name = Path(entry['config']).stem
            with (directory/(name+'.log')).open('w') as log:
                processes[name] = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                    '--worker',json.dumps(entry),'--directory',str(directory/name)],cwd=REPO,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return int(any(wait_runs(processes,directory).values()))
    finally:
        for process in processes.values():
            stop_process(process)
        for lock in locks:
            lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=REPO/'scripts/train/configs/abci/sft_two_nodes.json')
    parser.add_argument('--node')
    parser.add_argument('--run',action='store_true',help='Start configured runs inside an existing PBS allocation')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    parser.add_argument('--directory',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    if args.worker:
        return worker(json.loads(args.worker),args.directory)
    nodes = validate_plan(json.loads(args.plan.read_text()))
    if args.node and args.node not in nodes:
        parser.error('Unknown node name')
    if not args.run:
        print(json.dumps(nodes if not args.node else nodes[args.node],indent=2))
        return 0
    if not args.node:
        parser.error('--run requires --node')
    return launch_node(nodes[args.node],args.node)


if __name__ == '__main__':
    raise SystemExit(main())
