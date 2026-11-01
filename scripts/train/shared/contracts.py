"""Validate supported training settings before loading data or model weights."""

import math


def keys(value, required, where, optional=''):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be a mapping')
    missing = set(required.split()) - value.keys()
    unknown = value.keys() - set((required + ' ' + optional).split())
    if missing or unknown:
        raise ValueError(f'{where}: missing fields {sorted(missing)}; unsupported fields {sorted(unknown)}')


def fixed(value, expected, where):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f'{where} is fixed at {expected!r}; this implementation does not support {value!r}')


def number(value, where, *, minimum=0, maximum=None, integer=False, strict=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (integer and type(value) is not int)
            or value < minimum or (strict and value == minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{where} has an invalid numeric value: {value!r}')


def boolean(value, where):
    if type(value) is not bool:
        raise ValueError(f'{where} must be true or false')


# Accepted only for existing run snapshots. New templates omit fixed behavior.
RL_FIXED = {
    'policy': {'equal_reward_groups': 'retain_with_kl'},
    'training': {'gradient_checkpointing': True, 'base_dtype': 'bfloat16', 'adapter_dtype': 'float32'},
    'evaluation': {'before_training': True, 'save_on_evaluation': True,
                   'metric': 'medical_category_macro_f1', 'greater_is_better': True,
                   'starting_checkpoint_eligible': True},
    'rewards': {'medical_vqa': 'selected_label_f1', 'single_label': 'exact_match',
                'invalid_answer': 0.0, 'correct_negative': 1.0,
                'subject_localization': 'point_inside_subject_box',
                'counting': 'relative_absolute_error', 'camera_view': 'exact_match'},
    'rewards.spatial': {'mandatory_only': True, 'human_annotated_frames_only': True,
                        'matching': 'one_to_one', 'denominator': 'max_prediction_required_target_count'},
    'rewards.temporal': {'mandatory_only': True, 'require_positive_overlap': True,
                         'boundary_combination': 'product', 'matching': 'one_to_one',
                         'denominator': 'max_prediction_target_count'},
}
RL_FIELDS = {
    'policy': 'thinking disable_dropout importance_sampling_level epsilon epsilon_high beta sync_ref_model num_generations num_iterations temperature top_p top_k repetition_penalty max_completion_length reward_aggregation scale_rewards',
    'batch': 'questions_per_update questions_per_rollout responses_per_device',
    'training': 'num_train_epochs learning_rate lr_scheduler_type warmup_ratio optim weight_decay adam_beta1 adam_beta2 adam_epsilon max_grad_norm',
    'execution': 'backend cpu_offload vllm_mode rollout_gpus rollout_tensor_parallel_size',
    'evaluation': 'interval_epochs',
    'rewards': 'answer_and_spatial answer_and_temporal answer_and_both spatial temporal',
    'rewards.spatial': 'timestamp_tolerance_seconds native_full_credit_pixels native_zero_credit_pixels stabilized_full_credit_pixels stabilized_zero_credit_pixels label_policy',
    'rewards.temporal': 'start_full_seconds start_zero_seconds end_full_seconds end_zero_seconds',
}


def validate_rl_contract(config, *, resolved=False):
    keys(config, 'policy batch training execution evaluation rewards', 'rl',
         'recipe initialization experiment_config sft_config' if resolved else '')
    for path, required in RL_FIELDS.items():
        node = config
        for part in path.split('.'):
            node = node[part]
        constants = RL_FIXED.get(path, {})
        optional = ' '.join(constants)
        if resolved and path == 'execution':
            required += ' training_gpus'
        keys(node, required, 'rl.' + path, optional)
        for key, expected in constants.items():
            if key in node:
                fixed(node[key], expected, f'rl.{path}.{key}')
    for key, expected in {'thinking': False, 'disable_dropout': True, 'sync_ref_model': False}.items():
        fixed(config['policy'][key], expected, 'rl.policy.' + key)
    number(config['policy']['top_k'], 'rl.policy.top_k', integer=True)
    number(config['policy']['epsilon_high'], 'rl.policy.epsilon_high', maximum=1)
    fixed(config['execution']['cpu_offload'], False, 'rl.execution.cpu_offload')
    if config['training']['optim'] not in ('adamw_torch', 'adamw_torch_fused'):
        raise ValueError('rl.training.optim must be adamw_torch or adamw_torch_fused')
    rewards = config['rewards']
    for name, fields in (('answer_and_spatial', 'answer spatial'),
                         ('answer_and_temporal', 'answer temporal'),
                         ('answer_and_both', 'answer spatial temporal')):
        keys(rewards[name], fields, 'rl.rewards.' + name)
    policy = rewards['spatial']['label_policy']
    keys(policy, 'excluded_categories categories', 'spatial.label_policy')
    if not isinstance(policy['categories'], dict):
        raise ValueError('spatial.label_policy.categories must be a mapping')
    for category, rule in policy['categories'].items():
        keys(rule, 'mode', f'spatial.label_policy.categories.{category}',
             'labels full_credit_pixels zero_credit_pixels placement_instructions')
        instructions = rule.get('placement_instructions', {})
        if not isinstance(instructions, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                for k, v in instructions.items()):
            raise ValueError(f'{category}.placement_instructions must map label keys to text')
        if 'labels' in rule and not set(instructions).issubset(rule['labels']):
            raise ValueError(f'{category}: placement instructions must use allowed label keys')
    if resolved:
        keys(config['initialization'], 'source continue_adapter fresh_optimizer', 'rl.initialization')


LOADER_FIELDS = ('dataloader_num_workers dataloader_persistent_workers dataloader_prefetch_factor '
                 'dataloader_pin_memory dataloader_multiprocessing_context')
SFT_FIELDS = ('fsdp fsdp_config num_train_epochs learning_rate per_device_train_batch_size warmup_steps '
              'weight_decay adam_beta1 adam_beta2 adam_epsilon max_grad_norm lr_scheduler_type optim '
              'logging_steps eval_strategy save_strategy save_total_limit load_best_model_at_end '
              'metric_for_best_model greater_is_better accelerator_config')


def validate_experiment(config, *, sft=False):
    """Validate the flattened experiment used by both direct and run-YAML entry points."""
    keys(config, 'method attention max_length video augmentation lora data_checks evaluation training',
         'experiment', 'recipe effective_batch_size dataset mixture')
    fixed(config['method'], 'lora', 'method')
    if config['attention'] not in ('sdpa', 'eager', 'flash_attention_2', 'flash_attention_3'):
        raise ValueError('Unsupported attention implementation')
    number(config['max_length'], 'max_length', integer=True, strict=True)
    video = config['video']
    keys(video, 'fps qwen gemma', 'video')
    number(video['fps'], 'video.fps', strict=True)
    keys(video['qwen'], 'max_pixels', 'video.qwen')
    number(video['qwen']['max_pixels'], 'video.qwen.max_pixels', integer=True, strict=True)
    keys(video['gemma'], 'max_soft_tokens', 'video.gemma')
    if type(video['gemma']['max_soft_tokens']) is not int or video['gemma']['max_soft_tokens'] not in (70, 140, 280, 560, 1120):
        raise ValueError('video.gemma.max_soft_tokens must be 70, 140, 280, 560, or 1120')
    lora = config['lora']
    keys(lora, 'r lora_alpha lora_dropout bias use_rslora', 'lora')
    for key in ('r', 'lora_alpha'):
        number(lora[key], 'lora.' + key, integer=True, strict=True)
    number(lora['lora_dropout'], 'lora.lora_dropout', maximum=1)
    fixed(lora['bias'], 'none', 'lora.bias')
    boolean(lora['use_rslora'], 'lora.use_rslora')
    keys(config['data_checks'], 'stabilized_coordinates_verified', 'data_checks')
    boolean(config['data_checks']['stabilized_coordinates_verified'], 'stabilized_coordinates_verified')
    keys(config['evaluation'], 'max_new_tokens', 'evaluation')
    number(config['evaluation']['max_new_tokens'], 'evaluation.max_new_tokens', integer=True, strict=True)
    train = config['training']
    keys(train, 'seed ' + LOADER_FIELDS + (' ' + SFT_FIELDS if sft else ''), 'training',
         'data_seed per_device_eval_batch_size' if sft else 'data_seed')
    for key in ('seed', 'data_seed', 'dataloader_num_workers'):
        if key in train and not (key == 'data_seed' and train[key] is None):
            number(train[key], 'training.' + key, integer=True)
    for key in ('dataloader_persistent_workers', 'dataloader_pin_memory'):
        boolean(train[key], 'training.' + key)
    number(train['dataloader_prefetch_factor'], 'dataloader_prefetch_factor', integer=True, strict=True)
    if train['dataloader_multiprocessing_context'] not in ('spawn', 'forkserver', 'fork'):
        raise ValueError('Unsupported dataloader_multiprocessing_context')
    if train['dataloader_num_workers'] == 0 and train['dataloader_persistent_workers']:
        raise ValueError('Persistent workers require dataloader_num_workers > 0')
    if not sft:
        return
    if 'per_device_eval_batch_size' in train:
        fixed(train['per_device_eval_batch_size'], 1, 'training.per_device_eval_batch_size')
    number(config['effective_batch_size'], 'effective_batch_size', integer=True, strict=True)
    for key in ('per_device_train_batch_size', 'logging_steps', 'save_total_limit'):
        number(train[key], 'training.' + key, integer=True, strict=True)
    for key in ('num_train_epochs', 'learning_rate', 'adam_epsilon', 'max_grad_norm'):
        number(train[key], 'training.' + key, strict=True)
    for key in ('adam_beta1', 'adam_beta2'):
        number(train[key], 'training.' + key, maximum=1)
        if train[key] == 1:
            raise ValueError(f'{key} must be below 1')
    number(train['weight_decay'], 'training.weight_decay')
    number(train['warmup_steps'], 'training.warmup_steps')
    if train['warmup_steps'] >= 1 and type(train['warmup_steps']) is not int:
        raise ValueError('warmup_steps must be an integer count or a fraction below 1')
    fixed(train['metric_for_best_model'], 'macro_f1', 'training.metric_for_best_model')
    fixed(train['greater_is_better'], True, 'training.greater_is_better')
    boolean(train['load_best_model_at_end'], 'training.load_best_model_at_end')
    boolean(train['fsdp'], 'training.fsdp')
    if train['optim'] not in ('adamw_torch', 'adamw_torch_fused'):
        raise ValueError('training.optim must be adamw_torch or adamw_torch_fused')
    if train['lr_scheduler_type'] not in ('cosine', 'linear', 'constant', 'constant_with_warmup'):
        raise ValueError('Unsupported SFT learning-rate scheduler')
    for key in ('eval_strategy', 'save_strategy'):
        if train[key] not in ('epoch', 'no'):
            raise ValueError(f'{key} supports epoch or no; step intervals are not configured')
    if train['load_best_model_at_end'] and (train['eval_strategy'] != 'epoch' or train['save_strategy'] != 'epoch'):
        raise ValueError('Best-checkpoint selection requires epoch evaluation and saving')
    fsdp = train['fsdp_config']
    keys(fsdp, 'version reshard_after_forward activation_checkpointing cpu_offload state_dict_type', 'fsdp_config')
    fixed(fsdp['version'], 2, 'fsdp_config.version')
    for key in ('reshard_after_forward', 'activation_checkpointing', 'cpu_offload'):
        boolean(fsdp[key], 'fsdp_config.' + key)
    fixed(fsdp['state_dict_type'], 'FULL_STATE_DICT', 'fsdp_config.state_dict_type')
    keys(train['accelerator_config'], 'non_blocking', 'accelerator_config')
    boolean(train['accelerator_config']['non_blocking'], 'accelerator_config.non_blocking')


def validate_generation(settings):
    keys(settings, 'do_sample temperature top_p top_k min_p presence_penalty repetition_penalty', 'model.generation')
    boolean(settings['do_sample'], 'model.generation.do_sample')
    number(settings['temperature'], 'model.generation.temperature')
    if settings['do_sample'] and settings['temperature'] == 0:
        raise ValueError('Sampling requires a positive temperature; use do_sample: false for greedy decoding')
    number(settings['top_p'], 'model.generation.top_p', maximum=1, strict=True)
    number(settings['min_p'], 'model.generation.min_p', maximum=1)
    number(settings['top_k'], 'model.generation.top_k', integer=True)
    number(settings['presence_penalty'], 'model.generation.presence_penalty', minimum=-2, maximum=2)
    number(settings['repetition_penalty'], 'model.generation.repetition_penalty', strict=True)
