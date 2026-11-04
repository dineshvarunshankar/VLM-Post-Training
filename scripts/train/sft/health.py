"""Fixed-subset teacher diagnostics, independent of checkpoint selection.

The optional stop policy compares medical-value loss against the best checked
recoverable step. Legacy policies retain their advisory baseline warnings.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
import copy
import logging
import json
from pathlib import Path
import math
import random
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HealthDeteriorationStop(RuntimeError):
    """Training stopped deliberately; success export and test evaluation are invalid."""


class PeriodicHealthMixin:
    def configure_health(self, policy):
        """Configure after Trainer initialization; None preserves legacy behavior."""
        self.health_policy = None if policy is None else dict(policy)
        self._health_last_step = None
        self._health_baselines = {}
        self._health_above_baseline = 0
        self._health_eval_dataloaders = {}
        self.health_subset_indices = ()
        self._health_state = None
        if policy is None:
            return
        for key in ('every_n_steps', 'max_examples'):
            if type(policy.get(key)) is not int or policy[key] < 1:
                raise ValueError(f'Periodic health {key} must be a positive integer')
        if policy.get('stop_on_deterioration') is not None:
            from scripts.train.shared.contracts import validate_sft_reliability
            validate_sft_reliability({'health_check': policy,
                                     'recovery': getattr(self, '_recovery_policy', None)})
        dataset = self.eval_dataset
        if dataset is None or isinstance(dataset, dict):
            raise ValueError('Periodic health requires one validation dataset')
        cot = getattr(self.video_collator, 'answer_format', None) == 'cot'
        strata = defaultdict(deque)
        medical = eligible = 0
        for index in range(len(dataset)):
            row = dataset[index]
            reference = row.get('reference') or {}
            if row.get('task', reference.get('task', 'medical_vqa')) != 'medical_vqa':
                continue
            medical += 1
            if cot and not (row.get('teacher_available') and
                            isinstance(row.get('reasoning'), str) and row['reasoning'].strip()):
                continue
            eligible += 1
            key = (str(reference.get('variant', row.get('source', 'unknown'))),
                   str(reference.get('category', row.get('category', 'unknown'))))
            strata[key].append(index)
        selected = []
        selected_strata = set()
        while strata and len(selected) < policy['max_examples']:
            for key in sorted(strata):
                selected.append(strata[key].popleft())
                selected_strata.add(key)
                if not strata[key]:
                    del strata[key]
                if len(selected) == policy['max_examples']:
                    break
        self.health_subset_indices = tuple(selected)
        self._health_dataset = (dataset.select(selected) if hasattr(dataset, 'select')
                                else [dataset[index] for index in selected])
        self._health_coverage = {
            'health_teacher_loss_expected': medical,
            'health_teacher_loss_covered': eligible,
            'health_teacher_coverage': eligible / medical if medical else 0.,
            'health_examples': len(selected),
            'health_strata_sampled': len(selected_strata),
        }
        if policy.get('stop_on_deterioration') is not None:
            if not selected:
                raise ValueError('Deterioration protection requires eligible medical health examples')
            from scripts.train.shared.runtime import data_fingerprint
            self._health_state = {
                'schema_version': 1, 'policy': copy.deepcopy(policy),
                'subset_sha256': data_fingerprint(list(self._health_dataset)),
                'subset_indices': list(selected), 'last_checked_step': None,
                'best_step': None, 'best_value_loss': None, 'bad_checks': 0,
                'stopped': False, 'stop_reason': None,
            }

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        self.run_periodic_health()
        if self._health_stopped():
            self._apply_health_stop()
            return None
        return super()._maybe_log_save_evaluate(*args, **kwargs)

    def _health_stopped(self):
        return bool(getattr(self, '_health_state', None) and self._health_state['stopped'])

    def _apply_health_stop(self):
        self.control.should_training_stop = True
        self.control.should_evaluate = False
        self.control.should_save = False

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if getattr(self, '_health_state', None) is not None:
            if resume_from_checkpoint is True:
                raise ValueError('Deterioration protection requires an explicit resume checkpoint')
            if resume_from_checkpoint:
                self.restore_health_state(resume_from_checkpoint)
            if self._health_stopped():
                raise HealthDeteriorationStop('Medical health deterioration: stopped checkpoint cannot resume training')
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def _run_epoch(self, *args, **kwargs):
        # A recovery save precedes its health measurement. After an interrupted
        # check, complete that check on the loaded weights before any update.
        if getattr(self, '_health_state', None) is not None:
            self.run_periodic_health()
            if self._health_stopped():
                self._apply_health_stop()
                return None
        return super()._run_epoch(*args, **kwargs)

    def _finalize_training(self, *args, **kwargs):
        # An exact epoch-boundary resume can skip _run_epoch entirely.
        if getattr(self, '_health_state', None) is not None:
            self.run_periodic_health()
        if self._health_stopped():
            self.is_in_train = False
            raise HealthDeteriorationStop('Medical health deterioration: training stopped; no successful export or test completion')
        return super()._finalize_training(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        result = super()._save_checkpoint(model, trial)
        if getattr(self, '_health_state', None) is not None:
            from scripts.train.shared.data import atomic_json
            from scripts.train.sft.recovery import _save_phase
            checkpoint = Path(self.args.output_dir) / f'checkpoint-{self.state.global_step}'
            def write():
                if self.is_world_process_zero():
                    atomic_json(checkpoint / 'health_state.json', self._health_state)
            _save_phase(write)
        return result

    def restore_health_state(self, checkpoint):
        """Load this checkpoint's monitor state, never a later root-level baseline."""
        from scripts.train.sft.recovery import _collective_error
        restored, error = None, None
        try:
            checkpoint = Path(checkpoint)
            restored = json.loads((checkpoint / 'health_state.json').read_text())
            step = json.loads((checkpoint / 'trainer_state.json').read_text())['global_step']
            expected = self._health_state
            if not isinstance(restored, dict) or set(restored) != set(expected):
                raise ValueError('Missing or malformed checkpoint health policy state')
            if restored['schema_version'] != 1 or restored['policy'] != expected['policy']:
                raise ValueError('Checkpoint health policy differs from configured policy')
            if (restored['subset_sha256'] != expected['subset_sha256']
                    or restored['subset_indices'] != expected['subset_indices']):
                raise ValueError('Checkpoint health subset differs from configured subset')
            last, best = restored['last_checked_step'], restored['best_step']
            if ((last is not None and (type(last) is not int or not 0 < last <= step))
                    or (best is not None and (type(best) is not int or last is None or not 0 < best <= last))
                    or type(restored['bad_checks']) is not int or restored['bad_checks'] < 0
                    or type(restored['stopped']) is not bool):
                raise ValueError('Invalid checkpoint health step or confirmation state')
            value = restored['best_value_loss']
            if ((best is None) != (value is None)
                    or (value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0))):
                raise ValueError('Invalid checkpoint health baseline')
            if restored['stopped'] and not isinstance(restored['stop_reason'], dict):
                raise ValueError('Stopped checkpoint lacks deterioration reason')
            future = [path for path in Path(self.args.output_dir).glob('checkpoint-*')
                      if path.is_dir() and path.name.removeprefix('checkpoint-').isdigit()
                      and int(path.name.removeprefix('checkpoint-')) > step]
            if future:
                raise ValueError('Explicit resume would collide with future ordinary checkpoints; '
                                 'preserve them outside the active checkpoints directory before resuming')
        except Exception as exc:
            error = exc
        _collective_error(error)
        self.restore_health_recovery_state(restored, resume_step=step)
        self._health_state = restored
        self._health_last_step = restored['last_checked_step']

    @contextmanager
    def _health_isolation(self):
        """Keep diagnostics from consuming training randomness or callback state."""
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        modules = {}
        for root in (self.model, getattr(self, 'model_wrapped', None)):
            if root is not None:
                modules.update({module: module.training for module in root.modules()})
        control, control_values = self.control, copy.deepcopy(vars(self.control))
        handler = self.callback_handler
        # Evaluation invokes on_prediction_step. No health diagnostic should
        # trigger selection callbacks or alter a progress callback's state.
        isolated_handler = copy.copy(handler)
        isolated_handler.callbacks = []
        saved = {}
        for name in ('_metrics', '_eval_dataloaders', 'optimizer', 'gather_function'):
            if hasattr(self, name):
                saved[name] = getattr(self, name)
        try:
            self.callback_handler = isolated_handler
            self._eval_dataloaders = self._health_eval_dataloaders
            if '_metrics' in saved:
                self._metrics = copy.deepcopy(saved['_metrics'])
                for values in self._metrics.values():
                    values.clear()
            # Trainer's loss loop calls optimizer.eval() when present. Health
            # only reads model weights and must never switch optimizer mode.
            self.optimizer = None
            yield handler
        finally:
            self._health_eval_dataloaders = self._eval_dataloaders
            for name in ('_metrics', '_eval_dataloaders', 'optimizer', 'gather_function'):
                if name in saved:
                    setattr(self, name, saved[name])
                elif hasattr(self, name):
                    delattr(self, name)
            self.callback_handler = handler
            self.control = control
            vars(control).clear()
            vars(control).update(control_values)
            for module, training in modules.items():
                module.training = training
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def run_periodic_health(self):
        """All ranks run once per completed optimizer-step cadence."""
        policy = getattr(self, 'health_policy', None)
        if policy is None:
            return None
        if self._health_stopped():
            self._apply_health_stop()
            return None
        step = self.state.global_step
        if step <= 0 or step % policy['every_n_steps'] or step == self._health_last_step:
            return None
        self._health_last_step = step
        started = time.monotonic()
        metrics = dict(self._health_coverage, health_alert=0, health_invalid=0,
                       health_unavailable=int(not self.health_subset_indices))
        optimizer = getattr(self, 'optimizer', None)
        with self._health_isolation() as handler:
            if not self.health_subset_indices:
                logger.warning('Periodic health unavailable at step %s: no eligible medical teachers; training continues', step)
            else:
                output = self._teacher_loss_loop(
                    self.get_eval_dataloader(self._health_dataset),
                    description='Periodic medical teacher health',
                    prediction_loss_only=True, ignore_keys=None, metric_key_prefix='health')
                metrics.update({key: value for key, value in output.metrics.items()
                                if key == 'health_loss' or key.startswith('health_medical_')})
                self._health_trends(metrics, step)
            if self._health_state is not None:
                self._update_health_protection(metrics, step)
            metrics['health_runtime'] = time.monotonic() - started
            # Keep the real tracking callbacks, but don't merge or clear the
            # epoch/training TRL buffers with this independent diagnostic log.
            if hasattr(self, '_metrics'):
                for values in self._metrics.values():
                    values.clear()
            self.callback_handler = handler
            self.optimizer = optimizer
            self.log(metrics)
        if self._health_stopped():
            self._apply_health_stop()
        return metrics

    def _health_trends(self, metrics, step):
        required = ('health_loss', 'health_medical_value_loss',
                    'health_medical_value_accuracy', 'health_medical_value_tokens')
        invalid = [key for key in required if key not in metrics]
        for key, value in metrics.items():
            if key != 'health_loss' and not key.startswith('health_medical_'):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                invalid.append(key)
            elif ((key.endswith('_loss') and value < 0)
                  or (key.endswith('_accuracy') and not 0 <= value <= 1)
                  or (key.endswith('_tokens') and (value < 0 or int(value) != value))):
                invalid.append(key)
        tokens = metrics.get('health_medical_value_tokens', 0)
        if isinstance(tokens, (int, float)) and tokens <= 0:
            invalid.append('health_medical_value_tokens')
        if invalid:
            metrics['health_invalid'] = 1
            self._health_above_baseline = 0
            logger.warning('Periodic health invalid at step %s (%s); training continues',
                           step, ', '.join(sorted(set(invalid))))
        else:
            for key in ('health_loss', 'health_medical_value_loss'):
                baseline = self._health_baselines.setdefault(key, metrics[key])
                metrics[key + '_baseline'] = baseline
                metrics[key + '_baseline_delta'] = metrics[key] - baseline
                if baseline > 0:
                    metrics[key + '_baseline_ratio'] = metrics[key] / baseline
            value_key = 'health_medical_value_loss'
            self._health_above_baseline = (self._health_above_baseline + 1
                if metrics[value_key] > self._health_baselines[value_key] else 0)
            metrics['health_alert'] = int(self._health_above_baseline >= 2)
            if metrics['health_alert'] and self._health_state is None:
                logger.warning('Periodic health medical value loss is above its initial baseline '
                               'for %s consecutive checks at step %s; advisory trend only, training continues',
                               self._health_above_baseline, step)
        metrics['health_medical_value_loss_above_baseline_checks'] = self._health_above_baseline

    def _update_health_protection(self, metrics, step):
        """Persist one collective decision against the best recoverable check."""
        import torch.distributed as distributed
        from scripts.train.sft.recovery import _save_phase
        from scripts.train.shared.data import atomic_json

        state = copy.deepcopy(self._health_state)
        policy = self.health_policy['stop_on_deterioration']
        if not distributed.is_initialized() or distributed.get_rank() == 0:
            state['last_checked_step'] = step
            if metrics['health_invalid'] or metrics['health_unavailable']:
                state['bad_checks'] = 0
            else:
                value = metrics['health_medical_value_loss']
                best = state['best_value_loss']
                if best is None or value < best:
                    state.update(best_value_loss=value, best_step=step, bad_checks=0)
                else:
                    deteriorated = (value > best * policy['loss_factor']
                                    and value > best + policy['loss_delta'])
                    state['bad_checks'] = state['bad_checks'] + 1 if deteriorated else 0
                    if state['bad_checks'] >= policy['consecutive_checks']:
                        state['stopped'] = True
                        state['stop_reason'] = {
                            'kind': 'medical_value_loss_deterioration', 'global_step': step,
                            'value_loss': value, 'best_value_loss': best,
                            'best_step': state['best_step'], 'consecutive_checks': state['bad_checks'],
                            'loss_factor': policy['loss_factor'], 'loss_delta': policy['loss_delta'],
                        }
        if distributed.is_available() and distributed.is_initialized():
            payload = [state]
            distributed.broadcast_object_list(payload, src=0)
            state = payload[0]
        self.publish_health_state(state, pin_best=state['best_step'] == step)
        self._health_state = state
        metrics.update(health_stopped=int(state['stopped']),
                       health_deterioration_checks=state['bad_checks'],
                       health_alert=int(state['bad_checks'] > 0))
        if state['best_step'] is not None:
            metrics.update(health_best_checked_step=state['best_step'],
                           health_best_checked_value_loss=state['best_value_loss'])
        if state['stopped']:
            metrics['health_stop_reason'] = state['stop_reason']['kind']
            def write_stop():
                if self.is_world_process_zero():
                    atomic_json(Path(self.args.output_dir).parent / 'health_stop.json',
                                dict(status='stopped_deterioration', **state))
            _save_phase(write_stop)
            logger.error('Medical health deterioration confirmed at step %s; training stops, best checked step %s remains recoverable',
                         step, state['best_step'])
