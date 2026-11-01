#!/usr/bin/env python3
"""Inspect or launch an answer-only experiment from one run YAML."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from scripts.train.shared.run_config import load_run, compile_run
from scripts.train.shared.paths import REPO_ROOT


def public_config(config):
    # Resume is an invocation choice, not a change to the recorded experiment.
    return {k: v for k, v in config.items() if k not in ('resume', 'env_file')}


def bundle_path(plan):
    digest = hashlib.sha256(json.dumps(public_config(plan['run']), sort_keys=True).encode()).hexdigest()
    return Path(plan['run']['paths']['output_root']) / '.run-configs' / digest


def stage_command(plan, stage, bundle):
    c = plan['run']
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node', str(c['execution']['training_gpus']), stage['script']]
    options = {'config': str(bundle / stage['config']), 'models': str(bundle / 'models.yaml'),
               'model': c['model'], 'media-root': c['paths']['media_root'],
               'split-dir': c['paths']['split_dir'], 'catalog': c['paths']['catalog'],
               'prompts': c['paths']['prompts'], 'weights-root': c['paths']['weights_root'],
               'output-dir': stage['output'], 'run-name': c['run']['name'], 'run-date': c['run']['date'],
               'vllm-server-url': c['execution']['vllm_server_url'],
               'vllm-model-name': c['execution']['vllm_model_name']}
    if c['checkpoint']:
        options.update(checkpoint=c['checkpoint'], revision=c['model_settings']['revision'])
    if 'sft_run' in stage:
        options['sft-run'] = stage['sft_run']
    if stage.get('resume'):
        options['resume'] = stage['resume']
    for key, value in options.items():
        command.extend(['--' + key, str(value)])
    return command


def child_environment(plan, bundle):
    c = plan['run']
    env = os.environ.copy()
    if c['env_file']:
        from dotenv import dotenv_values
        for key, value in dotenv_values(c['env_file']).items():
            if value is not None:
                env.setdefault(key, value)
    # YAML controls settings; .env and the shell supply credentials.
    for key, value in c['cloud'].items():
        env['AIRLAB_' + key.upper()] = str(value)
    for key, value in c['wandb'].items():
        env['WANDB_' + key.upper()] = str(value)
    if c['execution']['training_devices'] is not None:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, c['execution']['training_devices']))
    env['TRIAGE_RUN_CONFIG'] = str(bundle / 'run.yaml')
    return env


def select_resume_stages(plan):
    checkpoint = plan['run']['resume']
    if not checkpoint:
        return
    previous_path = Path(checkpoint).parent.parent / 'run.json'
    previous = json.loads(previous_path.read_text())
    if previous.get('launch_config') != public_config(plan['run']):
        raise ValueError('Resume requires the original run YAML settings; change only resume')
    stage_name = previous['run_location']['recipe']
    index = next((i for i, stage in enumerate(plan['stages']) if stage['recipe'] == stage_name), None)
    if index is None:
        raise ValueError('Resume checkpoint stage is not part of this run')
    plan['stages'] = plan['stages'][index:]
    if Path(plan['stages'][0]['output']).resolve() != previous_path.parent.resolve():
        raise ValueError('Resume checkpoint must belong to the configured output directory')
    plan['stages'][0]['resume'] = checkpoint


def preflight(plan):
    c = plan['run']
    from scripts.train.shared.config import validate_grounding_readiness
    validate_grounding_readiness(plan['experiment'])
    for key in ('media_root', 'split_dir'):
        if not Path(c['paths'][key]).is_dir():
            raise ValueError(f'paths.{key} is not an existing directory')
    for key in ('catalog', 'prompts'):
        if not Path(c['paths'][key]).is_file():
            raise ValueError(f'paths.{key} is not an existing file')
    for name in ('train', 'validation', 'test'):
        if not (Path(c['paths']['split_dir']) / (name + '.jsonl')).is_file():
            raise ValueError(f'Missing {name}.jsonl in paths.split_dir')
    if c.get('sft_run') and not Path(c['sft_run']).is_dir():
        raise ValueError('sft_run is not an existing directory')
    if c['resume'] and not (Path(c['resume']) / 'trainer_state.json').is_file():
        raise ValueError('resume must name a Trainer checkpoint')
    for stage in plan['stages']:
        output = Path(stage['output'])
        if output.exists() and any(output.iterdir()) and not stage.get('resume'):
            raise ValueError(f'Output is nonempty: {output}; choose a new run or set resume')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('Run this launcher once; it starts torchrun itself')


def write_bundle(plan, directory):
    directory.mkdir(parents=True, exist_ok=True)
    documents = {'run.yaml': public_config(plan['run']), 'experiment.yaml': plan['experiment'],
                 'models.yaml': plan['models']}
    if plan['rl'] is not None:
        documents['rl.yaml'] = plan['rl']
    for name, value in documents.items():
        text = yaml.safe_dump(value, sort_keys=False)
        path = directory / name
        try:
            with path.open('x') as stream:
                stream.write(text)
        except FileExistsError:
            if path.read_text() != text:
                raise ValueError(f'Resolved config differs from existing bundle: {path}')


def launch(plan):
    select_resume_stages(plan)
    preflight(plan)
    bundle = bundle_path(plan)
    write_bundle(plan, bundle)
    env = child_environment(plan, bundle)
    for stage in plan['stages']:
        print(f"Starting {stage['recipe']}: {stage['output']}", flush=True)
        subprocess.run(stage_command(plan, stage, bundle), env=env, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true', help='Resolve settings without datasets, credentials, models, or writes')
    args = parser.parse_args()
    try:
        plan = compile_run(load_run(args.config))
        if args.inspect:
            select_resume_stages(plan)
            from scripts.train.shared.config import validate_grounding_readiness
            pending = []
            try:
                validate_grounding_readiness(plan['experiment'])
            except ValueError as error:
                pending.append(str(error))
            print(json.dumps({'run': plan['run'], 'stages': plan['stages'],
                              'commands': [stage_command(plan, s, bundle_path(plan)) for s in plan['stages']],
                              'rl_batch': plan['rl_batch'], 'inactive_tasks_for_selected_sources': plan['inactive_tasks'],
                              'pending_data_checks': pending,
                              'vllm': 'Start the compatible video vLLM server separately on its allocated GPUs.'}, indent=2))
        else:
            launch(plan)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
