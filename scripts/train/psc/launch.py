"""Two-node PSC SFT entrypoint, preserving the shared 16-GPU runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def validate_layout(run, scheduler='slurm'):
    execution = run['execution']
    large = ('qwen3.8-27b', 'gemma4-31b-it')
    small = ('qwen3.5-9b', 'gemma4-12b-it')
    allowed = large + small if scheduler == 'pbs' else large
    if scheduler not in ('pbs', 'slurm') or run['model'] not in allowed:
        raise ValueError('Model is unsupported by this scheduler entrypoint')
    ranks = 16 if run['model'] in large else 8
    if (execution['training_gpus'] != ranks or execution.get('training_devices') is not None
            or execution.get('vllm_gpu_sharing') is not True):
        raise ValueError(f'This model requires {ranks} training GPUs with shared evaluation')
    return ranks // 8


def evaluation_arguments(run, scheduler='slurm'):
    validate_layout(run, scheduler)
    from scripts.train.shared.cot_budget import server_arguments
    return server_arguments(run)



def verify_server_runtime(run):
    """Check installed endpoint patches without modifying a shared environment."""
    if run.get('answer_format') == 'cot' or run.get('subject_check', {}).get('enabled'):
        subprocess.run([sys.executable, '-m', 'scripts.train.rl.rollout.patch'], check=True)

def training_command(plan, stage_command, *, script, master, rank, job_id, python, scheduler='slurm'):
    nodes = validate_layout(plan['run'], scheduler)
    if rank not in tuple(str(i) for i in range(nodes)):
        raise ValueError('Expected one torchrun agent per allocated node')
    wrapper = []
    if (plan['run']['model'] == 'gemma4-31b-it'
            and plan['experiment']['attention'] == 'flex_attention'):
        wrapper = [str(Path(__file__).with_name('gemma_dynamic_flex_entry.py'))]
    return [python, '-m', 'torch.distributed.run', '--nnodes', str(nodes), '--nproc-per-node', '8',
            '--node-rank', rank, '--rdzv-id', job_id, '--master-addr', master,
            '--master-port', '29641', *wrapper, *stage_command[stage_command.index(script):]]


def scheduler_identity(scheduler, environment):
    """Map scheduler-provided node identity without changing the rank layout."""
    rank_key, job_key = ('TRIAGE_NODE_RANK', 'PBS_JOBID') if scheduler == 'pbs' else ('SLURM_PROCID', 'SLURM_JOB_ID')
    rank, job = environment.get(rank_key), environment.get(job_key)
    if scheduler not in ('pbs', 'slurm') or rank not in ('0', '1') or not job:
        raise ValueError('Expected a scheduler job ID and one of two node ranks')
    if not all(c.isalnum() or c in '.-_' for c in job):
        raise ValueError('Invalid scheduler job ID')
    return rank, job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'server-arguments'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--scheduler', choices=('slurm', 'pbs'), default='slurm')
    parser.add_argument('--job-root')
    parser.add_argument('--allocated-nodes', type=int)
    parser.add_argument('--master')
    parser.add_argument('--expected-model', choices=('qwen3.5-9b', 'qwen3.8-27b', 'gemma4-12b-it', 'gemma4-31b-it'))
    args = parser.parse_args()
    from scripts.train.shared.run_config import load_run, compile_run
    from scripts.train.run import preflight, write_bundle, child_environment, stage_command, select_resume_stages
    from scripts.train.shared.data import atomic_json
    if args.mode == 'server-arguments':
        flags = evaluation_arguments(load_run(args.config), args.scheduler)
        if flags:
            print('\n'.join(flags))
        return
    if not args.job_root or not args.master:
        parser.error('--job-root and --master are required')
    root = Path(args.job_root)
    if args.mode == 'prepare':
        from dotenv import dotenv_values
        from scripts.train.sft.launch import resolve_training_args, build_callbacks
        run = load_run(args.config)
        nodes = validate_layout(run, args.scheduler)
        if args.scheduler == 'pbs' and args.allocated_nodes != nodes:
            raise ValueError('PBS node allocation does not match the configured training ranks')
        if args.expected_model and run['model'] != args.expected_model:
            raise ValueError('Run model does not match the selected PSC batch script')
        run['execution']['vllm_server_url'] = f'http://{args.master}:8000'
        if not dotenv_values(run['env_file']).get('WANDB_API_KEY'):
            raise ValueError('Project .env needs your WANDB_API_KEY; shared-account login is not used')
        plan = compile_run(run)
        select_resume_stages(plan)
        preflight(plan)
        if len(plan['stages']) != 1 or plan['run']['recipe'] != 'sft':
            raise ValueError('PSC entrypoint requires one SFT stage')
        resolved = resolve_training_args(plan['experiment'], run['execution']['training_gpus'])
        if resolved['per_device_train_batch_size'] != 1 or resolved['gradient_accumulation_steps'] != 16 // run['execution']['training_gpus']:
            raise ValueError('PSC approved layout uses microbatch 1 and effective batch 16')
        build_callbacks(resolved)
        verify_server_runtime(run)
        from scripts.train.storage.cloud_models import verify_local_source
        model = plan['run']['model_settings']
        weights = (Path(plan['run']['checkpoint']) if plan['run'].get('checkpoint') else
                   Path(plan['run']['paths']['weights_root']) / model['repository'] / model['revision'])
        verify_local_source(weights.expanduser().resolve(), model, check_hashes=False)
        root.mkdir(parents=True, exist_ok=False)
        write_bundle(plan, root / 'bundle')
        atomic_json(root / 'plan.json', plan)
        atomic_json(root / 'allocation.json', {
            'reserved_gpus': nodes * 8, 'training_gpus': nodes * 8, 'evaluation_gpus': 2 if nodes == 2 else 1,
            'evaluation_placement': 'first node GPUs 4,5, shared sequentially with training' if nodes == 2 else 'GPU 4, shared sequentially with training',
            'idle_training_gpus': 0, 'server_arguments': evaluation_arguments(run, args.scheduler),
        })
        print('PSC_LAUNCH_PREFLIGHT_PASSED', flush=True)
    else:
        plan = json.loads((root / 'plan.json').read_text())
        stage = plan['stages'][0]
        rank, job_id = scheduler_identity(args.scheduler, os.environ)
        command = training_command(plan, stage_command(plan, stage, root / 'bundle'),
            script=stage['script'], master=args.master, rank=rank,
            job_id=job_id, python=sys.executable, scheduler=args.scheduler)
        env = child_environment(plan, root / 'bundle')
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        if plan['run']['model'].startswith('qwen'):
            cache = Path('/tmp') / f"triage-{os.getuid()}-{job_id}-qwen-train"
            for key, name in (('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('CUDA_CACHE_PATH', 'cuda')):
                path = cache / name
                path.mkdir(mode=0o700, parents=True, exist_ok=True)
                env[key] = str(path)
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
