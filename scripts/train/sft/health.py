"""Fixed-subset teacher diagnostics, independent of checkpoint selection.

Alerts are advisory: two consecutive medical-value losses above the first
finite baseline produce a warning. This is a trend indicator, not a statistical
significance test or a rule for stopping, saving, or resuming training.
"""
from collections import defaultdict, deque
from contextlib import contextmanager
import copy
import logging
import math
import random
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PeriodicHealthMixin:
    def configure_health(self, policy):
        """Configure after Trainer initialization; None preserves legacy behavior."""
        self.health_policy = None if policy is None else dict(policy)
        self._health_last_step = None
        self._health_baselines = {}
        self._health_above_baseline = 0
        self._health_eval_dataloaders = {}
        self.health_subset_indices = ()
        if policy is None:
            return
        for key in ('every_n_steps', 'max_examples'):
            if type(policy.get(key)) is not int or policy[key] < 1:
                raise ValueError(f'Periodic health {key} must be a positive integer')
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

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        self.run_periodic_health()
        return super()._maybe_log_save_evaluate(*args, **kwargs)

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
            metrics['health_runtime'] = time.monotonic() - started
            # Keep the real tracking callbacks, but don't merge or clear the
            # epoch/training TRL buffers with this independent diagnostic log.
            if hasattr(self, '_metrics'):
                for values in self._metrics.values():
                    values.clear()
            self.callback_handler = handler
            self.optimizer = optimizer
            self.log(metrics)
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
            if metrics['health_alert']:
                logger.warning('Periodic health medical value loss is above its initial baseline '
                               'for %s consecutive checks at step %s; advisory trend only, training continues',
                               self._health_above_baseline, step)
        metrics['health_medical_value_loss_above_baseline_checks'] = self._health_above_baseline
