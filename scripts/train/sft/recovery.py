"""Full Trainer recovery checkpoints, independent of validation selection."""

from copy import copy, deepcopy
import json
from pathlib import Path
import shutil


RECOVERY_COMPLETE = 'recovery_complete.json'


def _collective_error(error):
    """Publish filesystem/save failures before another rank enters evaluation."""
    import torch.distributed as distributed

    if distributed.is_available() and distributed.is_initialized():
        errors = [None] * distributed.get_world_size()
        distributed.all_gather_object(errors, None if error is None else str(error))
        if error is not None:
            raise error
        if any(message is not None for message in errors):
            raise RuntimeError(f'Recovery checkpoint failed on another rank: {errors}')
    elif error is not None:
        raise error


def _save_phase(operation, *args, **kwargs):
    """Finish each save phase collectively before entering the next one."""
    error = None
    try:
        operation(*args, **kwargs)
    except Exception as exc:
        error = exc
    _collective_error(error)


class RecoveryCheckpointMixin:
    def configure_recovery(self, policy):
        """Enable explicit recovery policy after Trainer initialization."""
        self._recovery_policy = None if policy is None else dict(policy)
        self._recovery_last_step = None
        self._recovery_saving = False

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        policy = getattr(self, '_recovery_policy', None)
        if policy and self.state.global_step > 0:
            periodic = self.state.global_step % policy['save_steps'] == 0
            before_evaluation = policy['before_evaluation'] and self.control.should_evaluate
            if periodic or before_evaluation:
                self._save_recovery_checkpoint()
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def save_recovery_before_evaluation(self):
        """Protect direct training-time evaluation, including an epoch boundary."""
        policy = getattr(self, '_recovery_policy', None)
        if policy and policy['before_evaluation']:
            self._save_recovery_checkpoint()

    def _prepare_recovery_directory(self, checkpoint):
        # Complete local filesystem setup before native state-dict collectives.
        checkpoint.mkdir(parents=True, exist_ok=True)

    def _save_recovery_contents(self, output_dir):
        """Keep native formats, with error agreement between collective phases.

        Calling TRL's whole _save_checkpoint is unsafe here: its rank-zero
        model-card write precedes a FLOPs collective. Likewise native Trainer
        groups FSDP model IO and subsequent optimizer collectives in one method.
        """
        if self.hp_search_backend is None:
            _save_phase(self.store_flos)
        _save_phase(self.save_model, output_dir, _internal_call=True)
        if self.is_fsdp_enabled:
            from transformers.trainer import (
                get_fsdp_ckpt_kwargs, save_fsdp_model, save_fsdp_optimizer,
            )

            # Accelerate adjusts FULL_STATE_DICT options during saving. Keep
            # those temporary changes off the live training plugin.
            plugin = copy(self.accelerator.state.fsdp_plugin)
            plugin.state_dict_config = copy(plugin.state_dict_config)
            _save_phase(save_fsdp_model, plugin, self.accelerator, self.model,
                        output_dir, **get_fsdp_ckpt_kwargs())
            _save_phase(save_fsdp_optimizer, plugin, self.accelerator,
                        self.optimizer, self.model, output_dir)
            _save_phase(self._save_recovery_scheduler, output_dir)
        else:
            _save_phase(self._save_optimizer_and_scheduler, output_dir)
        _save_phase(self._save_scaler, output_dir)
        _save_phase(self._save_rng_state, output_dir)
        _save_phase(self._save_recovery_trainer_state, output_dir)

    def _save_recovery_scheduler(self, output_dir):
        import torch
        from transformers.trainer import SCHEDULER_NAME

        if self.args.should_save:
            torch.save(self.lr_scheduler.state_dict(), Path(output_dir, SCHEDULER_NAME))

    def _save_recovery_trainer_state(self, output_dir):
        from transformers.trainer import ExportableState, TRAINER_STATE_NAME

        if self.args.should_save:
            for callback in self.callback_handler.callbacks + [self.control]:
                if isinstance(callback, ExportableState):
                    name = callback.__class__.__name__
                    callback_state = callback.state()
                    if isinstance(self.state.stateful_callbacks[name], list):
                        self.state.stateful_callbacks[name].append(callback_state)
                    else:
                        self.state.stateful_callbacks[name] = callback_state
            self.state.save_to_json(str(Path(output_dir, TRAINER_STATE_NAME)))

    def _save_recovery_checkpoint(self):
        policy = getattr(self, '_recovery_policy', None)
        step = self.state.global_step
        if (not policy or not self.is_in_train or step <= 0
                or self._recovery_saving or self._recovery_last_step == step):
            return
        if self.optimizer is None or self.lr_scheduler is None:
            raise RuntimeError('Recovery checkpoints require initialized optimizer and scheduler')

        # The normal trainer directory is run/checkpoints. Its sibling keeps the
        # same run.json relationship used by explicit --resume checkpoints.
        root = Path(self.args.output_dir).parent / 'recovery'
        staging = root / f'.incomplete-{step}'
        checkpoint = root / f'checkpoint-{step}'
        main_process = self.accelerator.is_main_process
        self._recovery_saving = True
        try:
            error = None
            already_complete = False
            if main_process:
                try:
                    if checkpoint.exists():
                        marker = checkpoint / RECOVERY_COMPLETE
                        metadata = json.loads(marker.read_text()) if marker.is_file() else None
                        if metadata != {'schema_version': 1, 'global_step': step, 'completed': True}:
                            raise ValueError(f'Recovery checkpoint is incomplete: {checkpoint}')
                        already_complete = True
                    # A failed attempt at this same step is safe to replace. No
                    # completed or ordinary checkpoint is removed here.
                    if not already_complete:
                        if staging.exists():
                            shutil.rmtree(staging)
                        staging.mkdir(parents=True)
                except Exception as exc:
                    error = exc
            _collective_error(error)
            import torch.distributed as distributed
            if distributed.is_available() and distributed.is_initialized():
                result = [already_complete]
                distributed.broadcast_object_list(result, src=0)
                already_complete = result[0]
            if already_complete:
                self._recovery_last_step = step
                return

            arguments, state, control = self.args, self.state, self.control
            current_flos = self.current_flos
            error = None
            try:
                # Native saves update stateful callbacks, FLOPs and best paths.
                # Save copies so recovery never changes selection or callbacks.
                self.args = copy(arguments)
                self.state = deepcopy(state)
                self.control = deepcopy(control)
                self.args.output_dir = str(staging)
                self.args.save_only_model = False
                self.args.save_strategy = 'no'
                self.args.save_total_limit = None
                self.args.push_to_hub = False
                self.args.save_on_each_node = False
                _save_phase(self._prepare_recovery_directory, staging / checkpoint.name)
                self._save_recovery_contents(str(staging / checkpoint.name))
            except Exception as exc:
                error = exc
            finally:
                self.args, self.state, self.control = arguments, state, control
                self.current_flos = current_flos
            _collective_error(error)

            error = None
            if main_process:
                try:
                    staged_checkpoint = staging / checkpoint.name
                    marker = staged_checkpoint / RECOVERY_COMPLETE
                    with marker.open('w') as stream:
                        json.dump({'schema_version': 1, 'global_step': step, 'completed': True}, stream)
                        stream.write('\n')
                    # Publish only after all ranks successfully saved. The
                    # directory rename exposes state and marker atomically.
                    staged_checkpoint.rename(checkpoint)
                    shutil.rmtree(staging)
                    completed = sorted(
                        (path for path in root.glob('checkpoint-*')
                         if path.name.removeprefix('checkpoint-').isdigit()
                         and (path / RECOVERY_COMPLETE).is_file()),
                        key=lambda path: int(path.name.removeprefix('checkpoint-')),
                    )
                    for old in completed[:-policy['save_total_limit']]:
                        shutil.rmtree(old)
                except Exception as exc:
                    error = exc
            _collective_error(error)
            self._recovery_last_step = step
        finally:
            self._recovery_saving = False
