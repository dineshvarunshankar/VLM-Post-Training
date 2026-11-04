"""Explicit SFT epoch extensions; original experiment records remain immutable."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'scheduler', 'start_learning_rate'}:
        raise ValueError('continuation requires scheduler and start_learning_rate')
    if policy['scheduler'] != 'cosine_tail':
        raise ValueError('continuation scheduler must be cosine_tail')
    lr = policy['start_learning_rate']
    if lr is not None and (type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0):
        raise ValueError('continuation start_learning_rate must be null or positive and finite')


def _normalize_training(previous, current):
    if previous.get('lr_scheduler_type') != 'cosine':
        raise ValueError('Continuation supports an original cosine schedule only')
    policy = current.get('continuation')
    validate_policy(policy)
    old_epochs, new_epochs = previous['num_train_epochs'], current['num_train_epochs']
    if (type(new_epochs) not in (int, float) or not math.isfinite(new_epochs)
            or new_epochs <= old_epochs):
        raise ValueError('Continuation requires an increased num_train_epochs')
    normalized = deepcopy(current)
    normalized.pop('continuation')
    normalized['num_train_epochs'] = old_epochs
    if normalized != previous:
        raise ValueError('Continuation may change only num_train_epochs and its explicit policy')
    return deepcopy(previous)


def validate_launch_extension(previous, current):
    if previous is None or previous.get('recipe') != 'sft' or current.get('recipe') != 'sft':
        raise ValueError('Continuation requires the original SFT launch configuration')
    normalized = deepcopy(current)
    normalized['sft']['training'] = _normalize_training(previous['sft']['training'], current['sft']['training'])
    # Scheduler placement changes the server hostname. All other execution settings are immutable.
    if 'execution' in previous and 'execution' in normalized:
        normalized['execution']['vllm_server_url'] = previous['execution']['vllm_server_url']
    if normalized != previous:
        raise ValueError('Continuation launch configuration differs beyond epochs, policy or server endpoint')


def validate_metadata_extension(previous, current):
    normalized = deepcopy(current)
    normalized['config']['training'] = _normalize_training(previous['config']['training'], current['config']['training'])
    normalized['training_arguments'] = _normalize_training(previous['training_arguments'], current['training_arguments'])
    if 'launch_config' in previous or 'launch_config' in current:
        validate_launch_extension(previous.get('launch_config'), current.get('launch_config', {}))
        normalized['launch_config'] = deepcopy(previous['launch_config'])
    if normalized != previous:
        raise ValueError('Continuation metadata, data, model, prompts or runtime differ from run.json')


def validate_completion_outputs(output):
    output = Path(output)
    names = ('merged', 'adapter', 'checkpoint_selection.json', 'train_metrics.json', 'test_results.json')
    present = [name for name in names if (output / name).exists()]
    if present:
        raise ValueError('Archive prior completion outputs before continuation: ' + ', '.join(present))


def validate_model_weights(checkpoint, *, fsdp=False):
    checkpoint = Path(checkpoint)
    names = ['adapter_model.safetensors', 'adapter_model.bin', 'model.safetensors', 'pytorch_model.bin']
    if fsdp:
        names.append('pytorch_model_fsdp.bin')
    if any((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in names):
        return
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = checkpoint / name
        if index.is_file():
            shards = set(json.loads(index.read_text()).get('weight_map', {}).values())
            if shards and all(Path(shard).name == shard and (checkpoint / shard).is_file()
                              and (checkpoint / shard).stat().st_size > 0 for shard in shards):
                return
    raise ValueError(f'Checkpoint model weights or indexed shards are missing: {checkpoint}')


def validate_checkpoint(checkpoint, *, fsdp=False, world_size=None, require_health=False,
                        require_weights=False, policy=None):
    from scripts.train.shared.runtime import validate_recovery_checkpoint
    checkpoint = Path(checkpoint)
    validate_recovery_checkpoint(checkpoint)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    step = state['global_step']
    if type(step) is not int or step <= 0:
        raise ValueError('Continuation requires a positive saved global_step')
    # Production FSDP uses FULL_STATE_DICT. Never accept a partial/sharded directory as a full optimizer.
    optimizer = checkpoint / ('optimizer.bin' if fsdp else 'optimizer.pt')
    if not optimizer.is_file() or not (checkpoint / 'scheduler.pt').is_file():
        raise ValueError('Continuation requires complete optimizer and scheduler checkpoint files')
    import torch
    scheduler = torch.load(checkpoint / 'scheduler.pt', map_location='cpu', weights_only=True)
    if policy is not None:
        validate_policy(policy)
        if ('continuation_schema' not in scheduler and policy['start_learning_rate'] is not None
                and state.get('max_steps') != step):
            raise ValueError('Explicit continuation LR requires a completed, exhausted schedule')
    if require_weights:
        validate_model_weights(checkpoint, fsdp=fsdp)
    if scheduler.get('last_epoch') != step:
        raise ValueError('Saved scheduler step differs from trainer global_step')
    if world_size is not None:
        names = ['rng_state.pth'] if world_size == 1 else [f'rng_state_{rank}.pth' for rank in range(world_size)]
        if any(not (checkpoint / name).is_file() for name in names):
            raise ValueError('Continuation requires saved RNG state for every training rank')
    best = state.get('best_model_checkpoint')
    if best and not (Path(best) / 'trainer_state.json').is_file():
        raise ValueError('Saved best checkpoint is missing')
    if best:
        validate_model_weights(best, fsdp=fsdp)
    health_path = checkpoint / 'health_state.json'
    if require_health and not health_path.is_file():
        raise ValueError('Continuation requires checkpoint-local health state')
    if health_path.exists():
        health = json.loads(health_path.read_text())
        if health.get('stopped'):
            raise ValueError('Health-stopped checkpoints cannot continue')
        best_step = health.get('best_step')
        if best_step is not None:
            best_path = checkpoint.parent.parent / 'recovery' / f'checkpoint-{best_step}'
            if not (best_path / 'trainer_state.json').is_file():
                raise ValueError('Saved best health checkpoint is missing')
            validate_recovery_checkpoint(best_path)
            if require_weights:
                validate_model_weights(best_path, fsdp=fsdp)
    return step


def save_continuation_metadata(previous_path, current, checkpoint):
    previous_path, checkpoint = Path(previous_path), Path(checkpoint).resolve()
    raw = previous_path.read_bytes()
    validate_metadata_extension(json.loads(raw), current)
    validate_completion_outputs(previous_path.parent)
    record = {'schema_version': 1, 'original_run_sha256': hashlib.sha256(raw).hexdigest(),
              'checkpoint': str(checkpoint),
              'checkpoint_trainer_state_sha256': hashlib.sha256((checkpoint / 'trainer_state.json').read_bytes()).hexdigest(),
              'metadata': current}
    scheduler = checkpoint / 'scheduler.pt'
    if scheduler.is_file():
        record['checkpoint_scheduler_sha256'] = hashlib.sha256(scheduler.read_bytes()).hexdigest()
    encoded = json.dumps(record, sort_keys=True).encode()
    destination = previous_path.parent / 'continuations' / (hashlib.sha256(encoded).hexdigest() + '.json')
    from scripts.train.shared.data import atomic_json
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError('Continuation provenance record differs')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, record)
    return record


def make_scheduler(optimizer, end_step, policy):
    from scripts.train.sft.continuation_scheduler import CosineTailScheduler
    return CosineTailScheduler(optimizer, end_step, policy)


class ContinuationMixin:
    """Keep the native checkpoint restoration path, replacing only its LR schedule."""
    continuation_policy = None

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if self.continuation_policy is not None:
            validate_policy(self.continuation_policy)
            if not resume_from_checkpoint or resume_from_checkpoint is True:
                raise ValueError('Continuation requires an explicit resume checkpoint')
            validate_completion_outputs(Path(self.args.output_dir).parent)
            self._continuation_resume_step = validate_checkpoint(
                resume_from_checkpoint, fsdp=self.is_fsdp_enabled, world_size=self.args.world_size,
                require_health=bool(getattr(self, "health_policy", None)),
                require_weights=True, policy=self.continuation_policy)
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.continuation_policy is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if num_training_steps <= self._continuation_resume_step:
            raise ValueError('Continuation horizon must follow saved global_step')
        if self.lr_scheduler is None:
            self.lr_scheduler = make_scheduler(optimizer or self.optimizer, num_training_steps, self.continuation_policy)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        if self.continuation_policy is not None:
            validate_checkpoint(checkpoint, fsdp=self.is_fsdp_enabled, world_size=self.args.world_size,
                                require_health=bool(getattr(self, "health_policy", None)),
                                require_weights=True, policy=self.continuation_policy)
        super()._load_optimizer_and_scheduler(checkpoint)
        if self.continuation_policy is not None:
            scheduler = self.lr_scheduler
            if not scheduler.restored or scheduler.last_epoch != self._continuation_resume_step:
                raise ValueError('Continuation scheduler step differs from saved global_step')
            self.continuation_schedule = scheduler.state_dict()
