#!/usr/bin/env python3
"""Inspect the agreed RL recipe without loading models or changing datasets."""


import argparse
import math
from copy import deepcopy
from pathlib import Path

from scripts.train.shared.paths import ANSWER_ONLY_CONFIG_DIR
from scripts.train.shared.config import load_training_config


def positive(value, name, integer=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")


def validate_recipe(recipe):
    if recipe['recipe'] not in ('rl_only', 'sft_then_rl'):
        raise ValueError('Expected rl_only or sft_then_rl recipe')
    initialization = recipe['initialization']
    expected = {
        'rl_only': ('cloud_base', False),
        'sft_then_rl': ('best_sft_checkpoint', True),
    }[recipe['recipe']]
    if (initialization['source'], initialization['continue_adapter']) != expected:
        raise ValueError('Starting weights and adapter policy do not match the recipe')
    if type(initialization['continue_adapter']) is not bool or initialization['fresh_optimizer'] is not True:
        raise ValueError('Use an explicit adapter policy and a fresh RL optimizer')
    if recipe['recipe'] == 'sft_then_rl' and not recipe.get('sft_config'):
        raise ValueError('sft_then_rl requires an SFT configuration')
    policy = recipe['policy']
    if policy['importance_sampling_level'] != 'sequence':
        raise ValueError('GSPO requires sequence-level importance ratios')
    if policy['thinking'] or not policy['disable_dropout']:
        raise ValueError('This recipe requires thinking and dropout disabled')
    if policy['reward_aggregation'] != 'sum_then_normalize':
        raise ValueError('This recipe combines rewards before group normalization')
    if policy['scale_rewards'] != 'group' or policy['sync_ref_model']:
        raise ValueError('Use group normalization and a frozen reference policy')
    for key in ('num_generations', 'num_iterations', 'max_completion_length'):
        positive(policy[key], key, integer=True)
    if policy['num_generations'] < 2:
        raise ValueError('Group advantages require at least two responses')
    for key in ('epsilon', 'epsilon_high', 'beta', 'temperature', 'top_p', 'repetition_penalty'):
        positive(policy[key], key)
    if policy['epsilon'] >= 1 or policy['top_p'] > 1:
        raise ValueError('epsilon must be below 1 and top_p at most 1')
    for key, value in recipe['batch'].items():
        if value is not None:
            positive(value, f'batch.{key}', integer=True)
    training = recipe['training']
    for key in ('learning_rate', 'num_train_epochs', 'adam_epsilon', 'max_grad_norm'):
        positive(training[key], key)
    for key in ('warmup_ratio', 'weight_decay', 'adam_beta1', 'adam_beta2'):
        value = training[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f'{key} must be finite and in [0, 1)')
    if training['lr_scheduler_type'] != 'constant_with_warmup':
        raise ValueError('This recipe uses constant learning rate after warmup')
    execution = recipe['execution']
    if execution['backend'] != 'fsdp2' or execution['cpu_offload']:
        raise ValueError('This recipe uses FSDP2 without CPU offload')
    if execution['vllm_mode'] != 'server':
        raise ValueError('This recipe uses separate vLLM server GPUs')
    for key in ('training_gpus', 'rollout_gpus'):
        positive(execution[key], key, integer=True)
    tp = execution['rollout_tensor_parallel_size']
    if tp is not None:
        positive(tp, 'rollout_tensor_parallel_size', integer=True)
        if execution['rollout_gpus'] % tp:
            raise ValueError('Rollout GPU count must be divisible by tensor parallel size')
    positive(recipe['evaluation']['interval_epochs'], 'interval_epochs')
    rewards = recipe['rewards']
    for key in ('answer_and_spatial', 'answer_and_temporal', 'answer_and_both'):
        values = rewards[key].values()
        for value in values:
            positive(value, key)
        if not math.isclose(sum(values), 1.0):
            raise ValueError(f'{key} reward weights must sum to 1')
    spatial = rewards['spatial']
    tolerance = spatial['timestamp_tolerance_seconds']
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('spatial.timestamp_tolerance_seconds must be finite and nonnegative')
    temporal = rewards['temporal']
    bounds = [
        (spatial['native_full_credit_pixels'], spatial['native_zero_credit_pixels'], 'native spatial'),
        (temporal['start_full_seconds'], temporal['start_zero_seconds'], 'span start'),
        (temporal['end_full_seconds'], temporal['end_zero_seconds'], 'span end'),
    ]
    policy = spatial.get('label_policy', {})
    excluded = policy.get('excluded_categories', [])
    if not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded):
        raise ValueError('Spatial excluded_categories must be a list of category keys')
    for category, rule in policy.get('categories', {}).items():
        if rule.get('mode', 'point') not in ('point', 'subject_box', 'disabled'):
            raise ValueError(f'Unknown spatial mode for {category}')
        if 'labels' in rule and (not isinstance(rule['labels'], list) or any(not isinstance(x, str) for x in rule['labels'])):
            raise ValueError(f'Invalid spatial label allowlist for {category}')
        if 'full_credit_pixels' in rule or 'zero_credit_pixels' in rule:
            bounds.append((rule.get('full_credit_pixels'), rule.get('zero_credit_pixels'), category))
    stable = (spatial['stabilized_full_credit_pixels'], spatial['stabilized_zero_credit_pixels'])
    if any(value is not None for value in stable):
        bounds.append((*stable, 'stabilized spatial'))
    for full, zero, name in bounds:
        positive(full, name)
        positive(zero, name)
        if full >= zero:
            raise ValueError(f'{name}: zero-credit boundary must exceed full-credit boundary')


def load_recipe(path):
    recipe = load_training_config(path)
    validate_recipe(recipe)
    return recipe


def resolve_model_recipe(recipe, model_profile):
    """Use the model's server allocation unless the recipe explicitly overrides it."""
    resolved = deepcopy(recipe)
    execution = resolved['execution']
    if execution['rollout_tensor_parallel_size'] is None:
        tp = model_profile.get('vllm_tensor_parallel_size')
        if tp is None:
            raise ValueError('Model profile needs vllm_tensor_parallel_size, or set the recipe override')
        execution['rollout_tensor_parallel_size'] = tp
    validate_recipe(resolved)
    return resolved


def resolve_batch(recipe):
    batch = recipe['batch']
    for name, value in batch.items():
        positive(value, name, integer=True)
    questions = batch['questions_per_update']
    rollout = batch['questions_per_rollout']
    responses = questions * recipe['policy']['num_generations']
    microbatch = batch['responses_per_device'] * recipe['execution']['training_gpus']
    if rollout % questions:
        raise ValueError('Questions per rollout must be divisible by questions per update')
    if responses % microbatch:
        raise ValueError('Responses per update must be divisible by the global response microbatch')
    return dict(
        responses_per_update=responses,
        responses_per_rollout=rollout * recipe['policy']['num_generations'],
        gradient_accumulation_steps=responses // microbatch,
        updates_per_rollout=rollout // questions,
    )


def inspect_recipe(path, expected_recipe=None, model_profile=None):
    recipe = load_recipe(path)
    if model_profile is not None:
        recipe = resolve_model_recipe(recipe, model_profile)
    if expected_recipe and recipe["recipe"] != expected_recipe:
        raise ValueError(f"This command requires {expected_recipe}, got {recipe['recipe']}")
    experiment_path = path.parent / recipe['experiment_config']
    experiment = load_training_config(experiment_path)
    if 'mixture' not in experiment:
        raise ValueError('experiment_config must specify the dataset and task mixture')
    print(f"Recipe: {recipe['recipe']} (configuration inspection; no training)")
    print(f'Experiment: {experiment_path.resolve()}')
    if recipe['recipe'] == 'rl_only':
        print('Stages: cloud base weights -> new LoRA adapter -> GSPO')
    else:
        sft_path = path.parent / recipe['sft_config']
        sft = load_training_config(sft_path)
        if sft.get('recipe') != 'sft_only':
            raise ValueError('sft_config must select an SFT-only configuration')
        if sft['lora'] != experiment['lora']:
            raise ValueError('SFT and RL must use the same adapter configuration')
        print('Stages: cloud base weights -> SFT -> best SFT adapter -> GSPO')
        print(f'SFT configuration: {sft_path.resolve()}')
    print('Sources: ' + ', '.join(k for k, v in experiment['dataset']['sources'].items() if v))
    tp = recipe['execution']['rollout_tensor_parallel_size']
    if tp is not None:
        print(f"vLLM: TP={tp}, DP={recipe['execution']['rollout_gpus'] // tp}; {recipe['execution']['rollout_gpus']} server GPUs")
    print(f"LoRA: rank {experiment['lora']['r']}, alpha {experiment['lora']['lora_alpha']}; RL dropout disabled")
    print(f"Responses per question: {recipe['policy']['num_generations']}")
    print(f"Learning rate: {recipe['training']['learning_rate']:g}; constant after {recipe['training']['warmup_ratio']:.0%} warmup")
    pending = [f'batch.{k}' for k, v in recipe['batch'].items() if v is None]
    if not pending:
        for key, value in resolve_batch(recipe).items():
            print(f'{key}: {value}')
    for section, key in (('execution', 'rollout_tensor_parallel_size'),):
        if recipe[section][key] is None:
            pending.append(f'{section}.{key}')
    if experiment['dataset']['sources']['stabilized'] and recipe['rewards']['spatial']['stabilized_full_credit_pixels'] is None:
        pending.append('stabilized evidence-coordinate alignment and tolerance')
    from scripts.train.shared.config import validate_grounding_readiness
    try:
        validate_grounding_readiness(experiment)
    except ValueError as error:
        pending.append(str(error))
    print('Pending: ' + '; '.join(pending) if pending else 'Recipe fields resolved')
    print('Launchers are connected; corrected-dataset and live GPU validation remain pending.')
    print('SFT optimizer and evaluation generation settings are not imported into this RL recipe.')


def inspect_dataset(args):
    from scripts.train.shared.data import load_splits, read_yaml
    from scripts.train.shared.media_geometry import collect_geometry
    from scripts.train.rl.data import build_rl_mixture
    from scripts.train.shared.config import select_medical_examples
    import json

    recipe = load_recipe(args.config)
    experiment = load_training_config(args.config.parent / recipe['experiment_config'])
    splits = select_medical_examples(load_splits(
        args.media_root,args.split_dir,args.catalog,args.prompts,
        sources=experiment['dataset']['sources']),experiment)
    geometry = collect_geometry(splits['train'],
        progress=lambda done,total: print(f'Verified training videos: {done}/{total}',flush=True))
    _,_,report = build_rl_mixture(splits['train'],read_yaml(args.catalog),
        read_yaml(args.prompts),experiment,recipe,geometry)
    summary = {k:v for k,v in report.items() if k not in ('excluded_candidates','reward_preparation_exclusions','task_prompt_hashes')}
    summary['annotation_exclusions'] = len(report['excluded_candidates'])
    summary['reward_exclusions'] = len(report['reward_preparation_exclusions'])
    summary['exclusion_preview'] = (report['excluded_candidates'] + report['reward_preparation_exclusions'])[:10]
    print(json.dumps(summary,indent=2))


def recipe_cli(expected_recipe=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = ANSWER_ONLY_CONFIG_DIR / 'recipes' / f'{expected_recipe}.yaml' if expected_recipe else None
    parser.add_argument('--config', type=Path, default=default, required=expected_recipe is None)
    parser.add_argument('--inspect', action='store_true', help='Inspect configuration; no training')
    for name in ('media-root','split-dir','catalog','prompts'):
        parser.add_argument('--'+name,type=Path)
    args = parser.parse_args()
    args.config = args.config.resolve()
    try:
        inspect_recipe(args.config.resolve(), expected_recipe)
        paths = [args.media_root,args.split_dir,args.catalog,args.prompts]
        if any(p is not None for p in paths):
            if not args.inspect or not all(p is not None for p in paths):
                raise ValueError('Dataset inspection requires --inspect, --media-root, --split-dir, --catalog, and --prompts')
            inspect_dataset(args)
        if expected_recipe and not args.inspect:
            parser.exit(2, 'Use train_rl.py or train_sft_then_rl.py to launch training.\n')
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f'Invalid recipe: {exc}\n')


if __name__ == '__main__':
    recipe_cli()
