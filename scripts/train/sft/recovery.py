"""Full Trainer recovery checkpoints, independent of validation selection."""

from copy import copy, deepcopy
import json
from pathlib import Path
import shutil
import tempfile


RECOVERY_COMPLETE = 'recovery_complete.json'
HEALTH_STATE = 'health_state.json'


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

    def health_recovery_checkpoint(self, step):
        """Return only a complete native checkpoint for the specified update."""
        if type(step) is not int or step <= 0:
            raise ValueError('A health checkpoint requires a positive optimizer step')
        checkpoint = Path(self.args.output_dir).parent / 'recovery' / f'checkpoint-{step}'
        try:
            marker = json.loads((checkpoint / RECOVERY_COMPLETE).read_text())
            saved = json.loads((checkpoint / 'trainer_state.json').read_text())
            if (marker != {'schema_version': 1, 'global_step': step, 'completed': True}
                    or type(saved.get('global_step')) is not int or saved['global_step'] != step):
                raise ValueError('Recovery checkpoint step does not match its health check')
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f'Health requires a completed recovery checkpoint at step {step}') from exc
        return checkpoint

    def _validate_health_pin(self, state):
        """Validate checkpoint identity; the health mixin validates its schema."""
        best = state.get('best_step')
        if best is not None:
            checked = state.get('last_checked_step')
            if type(checked) is not int or type(best) is not int or best > checked:
                raise ValueError('Best health checkpoint cannot follow the last health check')
            self.health_recovery_checkpoint(best)
        return best

    def _rotate_recovery_checkpoints(self, root):
        """Keep the latest rolling checkpoints plus the best checked update."""
        pinned = None
        state_file = root / HEALTH_STATE
        if state_file.is_file():
            pinned = self._validate_health_pin(json.loads(state_file.read_text()))
        completed = sorted(
            (path for path in root.glob('checkpoint-*')
             if path.name.removeprefix('checkpoint-').isdigit()
             and (path / RECOVERY_COMPLETE).is_file()),
            key=lambda path: int(path.name.removeprefix('checkpoint-')),
        )
        retained = set(completed[-self._recovery_policy['save_total_limit']:])
        if pinned is not None:
            retained.add(root / f'checkpoint-{pinned}')
        for old in completed:
            if old not in retained:
                shutil.rmtree(old)

    def publish_health_state(self, state, *, pin_best=False):
        """Attach a check to this update before atomically replacing the pin."""
        from scripts.train.shared.data import atomic_json

        def publish():
            if not self.accelerator.is_main_process:
                return
            step = self.state.global_step
            if type(state.get('last_checked_step')) is not int or state['last_checked_step'] != step:
                raise ValueError('Health results must describe the current optimizer step')
            checkpoint = self.health_recovery_checkpoint(step)
            best = self._validate_health_pin(state)
            root = checkpoint.parent
            if pin_best:
                if best != step:
                    raise ValueError('A new best health checkpoint must be the checked update')
            else:
                state_file = root / HEALTH_STATE
                previous = json.loads(state_file.read_text()) if state_file.exists() else {}
                if best != previous.get('best_step'):
                    raise ValueError('Changing the best health checkpoint requires pin_best=True')
            # A failed root write leaves the previous pin protected. A resume
            # from this completed checkpoint can still recover its own check.
            atomic_json(checkpoint / HEALTH_STATE, state)
            atomic_json(root / HEALTH_STATE, state)
            self._rotate_recovery_checkpoints(root)

        _save_phase(publish)

    def restore_health_recovery_state(self, state, *, resume_step):
        """Restore the caller's checkpoint-local history before any rotation."""
        from scripts.train.shared.data import atomic_json

        def restore():
            if self.accelerator.is_main_process:
                checked = state.get('last_checked_step')
                if (type(resume_step) is not int or resume_step < 0
                        or (checked is not None and (type(checked) is not int or checked > resume_step))):
                    raise ValueError('Health history cannot follow the explicit resume checkpoint')
                self._validate_health_pin(state)
                root = Path(self.args.output_dir).parent / 'recovery'
                future = sorted(path for path in root.glob('checkpoint-*')
                                if path.is_dir() and path.name.removeprefix('checkpoint-').isdigit()
                                and int(path.name.removeprefix('checkpoint-')) > resume_step)
                if future:
                    # Explicit rewind starts a new continuation. Keep old files
                    # intact, but remove them from active retention and reuse.
                    archive = Path(tempfile.mkdtemp(prefix='.superseded-', dir=root))
                    prior_state = root / HEALTH_STATE
                    if prior_state.is_file():
                        atomic_json(archive / HEALTH_STATE, json.loads(prior_state.read_text()))
                    for checkpoint in future:
                        checkpoint.rename(archive / checkpoint.name)
                atomic_json(root / HEALTH_STATE, state)

        _save_phase(restore)

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
                    from scripts.train.shared.data import atomic_json
                    if hasattr(self, '_health_state'):
                        health_state = self._health_state
                    else:
                        state_file = root / HEALTH_STATE
                        health_state = json.loads(state_file.read_text()) if state_file.is_file() else None
                    if health_state is not None:
                        atomic_json(staged_checkpoint / HEALTH_STATE, health_state)
                    marker = staged_checkpoint / RECOVERY_COMPLETE
                    with marker.open('w') as stream:
                        json.dump({'schema_version': 1, 'global_step': step, 'completed': True}, stream)
                        stream.write('\n')
                    # Publish only after all ranks successfully saved. The
                    # directory rename exposes state and marker atomically.
                    staged_checkpoint.rename(checkpoint)
                    shutil.rmtree(staging)
                    self._rotate_recovery_checkpoints(root)
                except Exception as exc:
                    error = exc
            _collective_error(error)
            self._recovery_last_step = step
        finally:
            self._recovery_saving = False
