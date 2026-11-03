"""Two-node PSC SFT entrypoint, preserving the shared 16-GPU runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def validate_layout(run):
    execution = run['execution']
    if run['model'] not in ('qwen3.8-27b', 'gemma4-31b-it'):
        raise ValueError('This PSC entrypoint supports Qwen 27B and Gemma 31B only')
    if (execution['training_gpus'] != 16 or execution.get('training_devices') is not None
            or execution.get('vllm_gpu_sharing') is not True):
        raise ValueError('PSC requires 16 training GPUs across two nodes with shared evaluation')


def evaluation_arguments(run):
    validate_layout(run)
    from scripts.train.shared.cot_budget import server_arguments
    return server_arguments(run)


def training_command(plan, stage_command, *, script, master, rank, job_id, python):
    validate_layout(plan['run'])
    if rank not in ('0', '1'):
        raise ValueError('Expected one torchrun agent per node, with node rank 0 or 1')
    wrapper = []
    if plan['run']['model'] == 'gemma4-31b-it':
        wrapper = [str(Path(__file__).with_name('gemma_dynamic_flex_entry.py'))]
    return [python, '-m', 'torch.distributed.run', '--nnodes', '2', '--nproc-per-node', '8',
            '--node-rank', rank, '--rdzv-id', job_id, '--master-addr', master,
            '--master-port', '29641', *wrapper, *stage_command[stage_command.index(script):]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train', 'server-arguments'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--job-root')
    parser.add_argument('--master')
    parser.add_argument('--expected-model', choices=('qwen3.8-27b', 'gemma4-31b-it'))
    args = parser.parse_args()
    from scripts.train.shared.run_config import load_run, compile_run
    from scripts.train.run import preflight, write_bundle, child_environment, stage_command
    from scripts.train.shared.data import atomic_json
    if args.mode == 'server-arguments':
        flags = evaluation_arguments(load_run(args.config))
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
        validate_layout(run)
        if args.expected_model and run['model'] != args.expected_model:
            raise ValueError('Run model does not match the selected PSC batch script')
        run['execution']['vllm_server_url'] = f'http://{args.master}:8000'
        if not dotenv_values(run['env_file']).get('WANDB_API_KEY'):
            raise ValueError('Project .env needs your WANDB_API_KEY; shared-account login is not used')
        plan = compile_run(run)
        preflight(plan)
        if len(plan['stages']) != 1 or plan['run']['recipe'] != 'sft':
            raise ValueError('PSC entrypoint requires one SFT stage')
        resolved = resolve_training_args(plan['experiment'], 16)
        if resolved['per_device_train_batch_size'] != 1 or resolved['gradient_accumulation_steps'] != 1:
            raise ValueError('PSC approved layout uses microbatch 1 and effective batch 16')
        build_callbacks(resolved)
        root.mkdir(parents=True, exist_ok=False)
        write_bundle(plan, root / 'bundle')
        atomic_json(root / 'plan.json', plan)
        atomic_json(root / 'allocation.json', {
            'reserved_gpus': 16, 'training_gpus': 16, 'evaluation_gpus': 2,
            'evaluation_placement': 'first node GPUs 4,5, shared sequentially with training',
            'idle_training_gpus': 0, 'server_arguments': evaluation_arguments(run),
        })
        print('PSC_LAUNCH_PREFLIGHT_PASSED', flush=True)
    else:
        plan = json.loads((root / 'plan.json').read_text())
        stage = plan['stages'][0]
        command = training_command(plan, stage_command(plan, stage, root / 'bundle'),
            script=stage['script'], master=args.master, rank=os.environ['SLURM_PROCID'],
            job_id=os.environ['SLURM_JOB_ID'], python=sys.executable)
        env = child_environment(plan, root / 'bundle')
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
        if plan['run']['model'] == 'qwen3.8-27b':
            cache = Path('/tmp') / f"triage-{os.getuid()}-{os.environ['SLURM_JOB_ID']}-qwen-train"
            for key, name in (('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('CUDA_CACHE_PATH', 'cuda')):
                path = cache / name
                path.mkdir(mode=0o700, parents=True, exist_ok=True)
                env[key] = str(path)
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
