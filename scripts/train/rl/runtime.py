"""Resolve RL arguments and verify the starting SFT artifact without GPU work."""

import json
import math
from pathlib import Path

from scripts.train.shared.data import sha256
from scripts.train.rl.recipe import resolve_batch


def training_arguments(recipe, experiment, questions, output, server_url):
    batch = resolve_batch(recipe)
    tp = recipe['execution']['rollout_tensor_parallel_size']
    if tp is None:
        raise ValueError('Set execution.rollout_tensor_parallel_size before launch')
    if type(questions) is not int or questions < 1:
        raise ValueError('RL training needs at least one eligible question')
    steps_per_epoch = math.ceil(questions / recipe['batch']['questions_per_rollout']) * batch['updates_per_rollout']
    interval = max(1, math.ceil(steps_per_epoch * recipe['evaluation']['interval_epochs']))
    policy, train = recipe['policy'], recipe['training']
    args = {key: train[key] for key in (
        'num_train_epochs', 'learning_rate', 'lr_scheduler_type',
        'optim', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm')}
    args.update({key: policy[key] for key in (
        'num_generations', 'num_iterations', 'temperature', 'top_p', 'top_k',
        'repetition_penalty', 'max_completion_length', 'importance_sampling_level',
        'epsilon', 'epsilon_high', 'beta', 'sync_ref_model', 'scale_rewards', 'disable_dropout')})
    args.update(
        output_dir=str(Path(output) / 'checkpoints'), warmup_steps=train['warmup_ratio'],
        per_device_train_batch_size=recipe['batch']['responses_per_device'],
        per_device_eval_batch_size=1, num_generations_eval=1,
        gradient_accumulation_steps=batch['gradient_accumulation_steps'],
        generation_batch_size=batch['responses_per_rollout'],
        loss_type='grpo', reward_weights=[1.0],
        use_vllm=True, vllm_mode='server', vllm_server_base_url=server_url,
        bf16=True, fsdp=True,
        fsdp_config={'version': 2, 'reshard_after_forward': True,
                     'activation_checkpointing': True, 'cpu_offload': False,
                     'state_dict_type': 'FULL_STATE_DICT'},
        gradient_checkpointing=False,
        remove_unused_columns=False, report_to='none',
        eval_strategy='steps', save_strategy='steps', eval_steps=interval, save_steps=interval,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model='macro_f1', greater_is_better=True,
        logging_steps=1, seed=experiment['training']['seed'],
        data_seed=(experiment['training']['data_seed']
                   if experiment['training'].get('data_seed') is not None
                   else experiment['training']['seed']),
        dataloader_num_workers=experiment['training'].get('dataloader_num_workers', 4),
        dataloader_pin_memory=experiment['training'].get('dataloader_pin_memory', True),
        dataloader_drop_last=False,
    )
    if args['dataloader_num_workers'] > 0:
        args.update(
            dataloader_persistent_workers=experiment['training'].get('dataloader_persistent_workers', True),
            dataloader_prefetch_factor=experiment['training'].get('dataloader_prefetch_factor', 1),
            dataloader_multiprocessing_context=experiment['training'].get('dataloader_multiprocessing_context', 'spawn'),
        )
    return args


def validate_sft_start(run, model_profile, experiment, catalog, prompts, split_dir=None, *, expected_sft_config=None):
    run = Path(run).resolve()
    metadata = json.loads((run / 'run.json').read_text())
    selection = json.loads((run / 'checkpoint_selection.json').read_text())
    if metadata['config'].get('recipe') != 'sft_only':
        raise ValueError('The starting run must be an SFT-only run')
    for key in ('repository', 'revision'):
        if metadata['model'][key] != model_profile[key]:
            raise ValueError(f'SFT base model {key} differs from the selected RL model')
    for key in ('lora', 'video', 'max_length'):
        if metadata['config'][key] != experiment[key]:
            raise ValueError(f'SFT and RL {key} settings differ')
    expected = experiment if expected_sft_config is None else expected_sft_config
    for key in ('dataset', 'mixture'):
        if metadata['config'].get(key) != expected.get(key):
            raise ValueError(f'Starting SFT {key} differs from the configured SFT recipe')
    if metadata['catalog_sha256'] != sha256(catalog) or metadata['prompts_sha256'] != sha256(prompts):
        raise ValueError('SFT and RL catalog or medical prompts differ')
    if split_dir is not None:
        for name, digest in metadata['split_sha256'].items():
            if sha256(Path(split_dir) / f'{name}.jsonl') != digest:
                raise ValueError('SFT and RL split files differ')
    if selection.get('best_score') is None or not selection.get('best_checkpoint'):
        raise ValueError('SFT run lacks a selected best checkpoint')
    adapter = run / 'adapter'
    if not (adapter / 'adapter_config.json').is_file() or not (adapter / 'adapter_model.safetensors').is_file():
        raise ValueError('SFT run lacks its selected adapter weights')
    # A verified merged export binds the selected adapter to its recorded hashes.
    from scripts.train.storage.export_model import verify_export
    verify_export(run)
    return adapter
