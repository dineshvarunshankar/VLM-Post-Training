"""Approved dual-threshold health stops preserve recoverability and block success."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Trainer, TrainingArguments

from scripts.train.sft.health import PeriodicHealthMixin
from scripts.train.sft.recovery import RecoveryCheckpointMixin
from scripts.train.tests import test_periodic_health as health_fixtures
from scripts.train.tests.test_sft_recovery import TinyModel


POLICY = {'every_n_steps': 1, 'max_examples': 2,
          'stop_on_deterioration': {'loss_factor': 2., 'loss_delta': .5, 'consecutive_checks': 2}}


def diagnostic_trainer(root):
    trainer = health_fixtures.PeriodicHealthTests().trainer()
    trainer.args = SimpleNamespace(output_dir=str(Path(root, 'checkpoints')), report_to=[])
    trainer.accelerator = SimpleNamespace(is_main_process=True)
    trainer._recovery_policy = {'before_evaluation': True, 'save_steps': 1, 'save_total_limit': 2}
    trainer.is_world_process_zero = lambda: True
    trainer.published = []
    trainer.health_recovery_checkpoint = lambda step: Path(root, 'recovery', f'checkpoint-{step}')
    trainer.publish_health_state = lambda state, **kwargs: trainer.published.append(copy.deepcopy(state))
    trainer.restore_health_recovery_state = lambda state, **kwargs: None
    trainer.configure_health(copy.deepcopy(POLICY))
    return trainer


def check(trainer, step, value):
    trainer.state.global_step = step
    trainer._teacher_loss_loop = lambda *a, **kw: SimpleNamespace(metrics={
        'health_loss': 2., 'health_medical_value_loss': value,
        'health_medical_value_tokens': 2, 'health_medical_value_accuracy': .5})
    return trainer.run_periodic_health()


def collective_health_worker(rank, port, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        trainer = diagnostic_trainer(root)
        trainer.accelerator.is_main_process = rank == 0
        trainer.is_world_process_zero = lambda: rank == 0
        for step, value in enumerate((.2, 1., 1.), 1):
            # Teacher metrics are normally globally aggregated. Differing local
            # inputs prove that the final stop decision comes from one rank.
            check(trainer, step, value if rank == 0 else .2)
        Path(root, f'rank-{rank}.json').write_text(json.dumps({
            'stopped': trainer.control.should_training_stop,
            'state': getattr(trainer, '_health_state', None)}))
    finally:
        dist.destroy_process_group()


class TinyHealthTrainer(RecoveryCheckpointMixin, PeriodicHealthMixin, Trainer):
    interrupt_health_step = None

    def _teacher_loss_loop(self, *args, **kwargs):
        if self.state.global_step == self.interrupt_health_step:
            raise RuntimeError('health interrupted before measurement')
        result = self.evaluation_loop(*args, **kwargs)
        value = getattr(self, 'health_values', {}).get(self.state.global_step, .2 if self.state.global_step == 1 else .9)
        result.metrics.update(health_medical_value_loss=value,
            health_medical_value_accuracy=.5, health_medical_value_tokens=1)
        return result


def native_trainer(root, max_steps):
    trainer = TinyHealthTrainer(model=TinyModel(), args=TrainingArguments(
        output_dir=str(Path(root, 'checkpoints')), use_cpu=True, report_to='none',
        max_steps=max_steps, per_device_train_batch_size=1, per_device_eval_batch_size=1,
        disable_tqdm=True, save_strategy='no', eval_strategy='no'),
        train_dataset=[{'input_ids':torch.ones(1), 'labels':torch.zeros(1)}]*8,
        eval_dataset=[{'input_ids':[1.], 'labels':[0.],
                       'reference':{'variant':'native','category':'injury'}}])
    trainer.video_collator = SimpleNamespace(answer_format='answer_only')
    trainer.configure_recovery({'save_steps':1,'save_total_limit':2,'before_evaluation':True})
    trainer.configure_health(copy.deepcopy(POLICY))
    return trainer


class HealthDeteriorationTests(unittest.TestCase):
    def test_both_strict_thresholds_and_consecutive_reset(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            # .7 equals best+.5; .6 exceeds factor but not absolute delta.
            for step, value in enumerate((.2, .7, .6, .8, .3, .8, .9), 1):
                metrics = check(trainer, step, value)
                self.assertEqual(trainer.control.should_training_stop, step == 7)
            state = trainer._health_state
            self.assertEqual(state['bad_checks'], 2)
            self.assertEqual(state['best_value_loss'], .2)
            self.assertEqual(state['best_step'], 1)
            self.assertEqual(state['stop_reason']['global_step'], 7)
            self.assertEqual(metrics['health_stopped'], 1)
            self.assertEqual(trainer.state.best_metric, .8)
            self.assertEqual(trainer.state.best_model_checkpoint, 'epoch-1')
            self.assertFalse(trainer.control.should_evaluate)
            self.assertFalse(trainer.control.should_save)
            self.assertTrue(Path(root, 'health_stop.json').is_file())

    def test_new_best_resets_counter_and_factor_equality_is_not_deterioration(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            for step, value in enumerate((1., 2., 2.01, .5, 1., 1.01), 1):
                check(trainer, step, value)
            self.assertFalse(trainer.control.should_training_stop)
            self.assertEqual(trainer._health_state['best_step'], 4)
            self.assertEqual(trainer._health_state['best_value_loss'], .5)
            self.assertEqual(trainer._health_state['bad_checks'], 1)

    def test_invalid_health_breaks_consecutive_finite_checks(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            for step, value in enumerate((.2, .9, float('nan'), .9), 1):
                check(trainer, step, value)
            self.assertFalse(trainer.control.should_training_stop)
            self.assertEqual(trainer._health_state['bad_checks'], 1)

    def test_resume_restores_checkpoint_state_and_rejects_different_subset(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            check(trainer, 1, .2); check(trainer, 2, .9)
            checkpoint = Path(root, 'recovery/checkpoint-2'); checkpoint.mkdir(parents=True)
            (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step': 2}))
            (checkpoint/'health_state.json').write_text(json.dumps(trainer._health_state))
            resumed = diagnostic_trainer(root)
            resumed.restore_health_state(checkpoint)
            check(resumed, 3, .9)
            self.assertTrue(resumed.control.should_training_stop)
            changed = diagnostic_trainer(root)
            changed.eval_dataset[0]['answer'] = '{"changed":1}'
            changed.configure_health(copy.deepcopy(POLICY))
            with self.assertRaisesRegex(ValueError, 'subset'):
                changed.restore_health_state(checkpoint)

    def test_all_ranks_share_stop_reason(self):
        with tempfile.TemporaryDirectory() as root:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            mp.spawn(collective_health_worker, args=(port, root), nprocs=2, join=True)
            rows = [json.loads(Path(root, f'rank-{rank}.json').read_text()) for rank in range(2)]
            self.assertTrue(all(row['stopped'] for row in rows))
            self.assertEqual(rows[0]['state'], rows[1]['state'])

    def test_stopped_run_cannot_save_adapter_evaluate_test_or_export(self):
        from scripts.train.sft.launch import save_training_outputs
        from scripts.train.shared.runtime import evaluate_heldout, finish_training_and_export
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            for step, value in enumerate((.2, .9, .9), 1):
                check(trainer, step, value)
            trainer.save_model = Mock(side_effect=AssertionError('success adapter save ran'))
            trainer.evaluate = Mock(side_effect=AssertionError('held-out generation ran'))
            trainer.accelerator.wait_for_everyone = Mock(side_effect=AssertionError('success export started'))
            for operation in (
                lambda: save_training_outputs(trainer, None, Path(root), None),
                lambda: evaluate_heldout(trainer, None, Path(root)),
                lambda: finish_training_and_export(trainer, Path(root)),
            ):
                with self.assertRaisesRegex(RuntimeError, 'deterioration'):
                    operation()
            for name in ('adapter', 'train_metrics.json', 'checkpoint_selection.json', 'test_results.json', 'merged'):
                self.assertFalse(Path(root, name).exists())

    def test_real_trainer_stops_updates_and_pins_best_checked_recovery(self):
        class HealthTrainer(RecoveryCheckpointMixin, PeriodicHealthMixin, Trainer):
            def _teacher_loss_loop(self, *args, **kwargs):
                result = self.evaluation_loop(*args, **kwargs)
                result.metrics.update(health_medical_value_loss={1:.2, 2:.9, 3:1.}[self.state.global_step],
                    health_medical_value_accuracy=.5, health_medical_value_tokens=1)
                return result
            def evaluate(self, *args, **kwargs):
                raise AssertionError('full generation entered on stop')
        with tempfile.TemporaryDirectory() as root:
            trainer = HealthTrainer(model=TinyModel(), args=TrainingArguments(
                output_dir=str(Path(root, 'checkpoints')), use_cpu=True, report_to='none',
                max_steps=5, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                disable_tqdm=True, save_strategy='no', eval_strategy='steps', eval_steps=3),
                train_dataset=[{'input_ids':torch.ones(1), 'labels':torch.zeros(1)}]*8,
                eval_dataset=[{'input_ids':[1.], 'labels':[0.],
                               'reference':{'variant':'native','category':'injury'}}])
            trainer.video_collator = SimpleNamespace(answer_format='answer_only')
            trainer.configure_recovery({'save_steps':1,'save_total_limit':2,'before_evaluation':True})
            trainer.configure_health(copy.deepcopy(POLICY))
            with self.assertRaisesRegex(RuntimeError, 'deterioration'):
                trainer.train()
            self.assertEqual(trainer.state.global_step, 3)
            for step in (1, 2, 3):
                self.assertTrue(Path(root, f'recovery/checkpoint-{step}/recovery_complete.json').is_file())
            saved = json.loads(Path(root, 'recovery/checkpoint-3/health_state.json').read_text())
            self.assertTrue(saved['stopped'])
            self.assertEqual(saved['best_step'], 1)
            self.assertIsNone(trainer.state.best_metric)
            self.assertFalse(trainer.is_in_train)
            self.assertFalse(any('train_runtime' in row for row in trainer.state.log_history))

    def test_full_trainer_resume_keeps_first_breach_through_native_state_reload(self):
        with tempfile.TemporaryDirectory() as root:
            first = native_trainer(root, 2)
            first.train()
            checkpoint = Path(root, 'recovery/checkpoint-2')
            resumed = native_trainer(root, 5)
            with self.assertRaisesRegex(RuntimeError, 'deterioration'):
                resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 3)
            self.assertEqual(resumed._health_state['bad_checks'], 2)
            self.assertEqual(resumed._health_state['best_step'], 1)

    def test_resume_runs_unfinished_due_health_check_before_any_update(self):
        with tempfile.TemporaryDirectory() as root:
            first = native_trainer(root, 5)
            first.interrupt_health_step = 3
            with self.assertRaisesRegex(RuntimeError, 'health interrupted'):
                first.train()
            checkpoint = Path(root, 'recovery/checkpoint-3')
            self.assertEqual(json.loads((checkpoint/'health_state.json').read_text())['last_checked_step'], 2)
            resumed = native_trainer(root, 5)
            with self.assertRaisesRegex(RuntimeError, 'deterioration'):
                resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 3)
            self.assertEqual(resumed._health_state['last_checked_step'], 3)

    def test_completed_epoch_resume_checks_pending_health_before_success(self):
        with tempfile.TemporaryDirectory() as root:
            first = native_trainer(root, 8)
            first.health_values = {step: .2 for step in range(1, 7)}
            first.interrupt_health_step = 8
            with self.assertRaisesRegex(RuntimeError, 'health interrupted'):
                first.train()
            checkpoint = Path(root, 'recovery/checkpoint-8')
            resumed = native_trainer(root, 8)
            with self.assertRaisesRegex(RuntimeError, 'deterioration'):
                resumed.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(resumed.state.global_step, 8)
            self.assertEqual(resumed._health_state['last_checked_step'], 8)
            self.assertFalse(any('train_runtime' in row for row in resumed.state.log_history))

    def test_stopped_checkpoint_resume_never_enters_training_updates(self):
        with tempfile.TemporaryDirectory() as root:
            first = native_trainer(root, 5)
            with self.assertRaisesRegex(RuntimeError, 'deterioration'):
                first.train()
            resumed = native_trainer(root, 5)
            resumed.training_step = Mock(side_effect=AssertionError('stopped checkpoint updated'))
            with self.assertRaisesRegex(RuntimeError, 'stopped checkpoint'):
                resumed.train(resume_from_checkpoint=str(Path(root, 'recovery/checkpoint-3')))
            self.assertEqual(resumed.state.global_step, 0)

    def test_future_ordinary_checkpoint_rejected_before_recovery_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            check(trainer, 1, .2)
            checkpoint = Path(root, 'recovery/checkpoint-1'); checkpoint.mkdir(parents=True)
            (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':1}))
            (checkpoint/'health_state.json').write_text(json.dumps(trainer._health_state))
            future = Path(root, 'checkpoints/checkpoint-2'); future.mkdir(parents=True)
            (future/'preserved').write_text('later original checkpoint')
            resumed = diagnostic_trainer(root)
            resumed.restore_health_recovery_state = Mock(side_effect=AssertionError('recovery mutated before collision validation'))
            with self.assertRaisesRegex(ValueError, 'future ordinary checkpoints'):
                resumed.restore_health_state(checkpoint)
            self.assertEqual((future/'preserved').read_text(), 'later original checkpoint')

    def test_tracking_records_stopped_reason_and_nonzero_finish(self):
        from scripts.train.shared.runtime import finish_deteriorated_training
        with tempfile.TemporaryDirectory() as root:
            trainer = diagnostic_trainer(root)
            for step, value in enumerate((.2, .9, .9), 1):
                check(trainer, step, value)
            trainer.args.report_to = ['wandb']
            closed = []
            tracker = SimpleNamespace(run=object(), summary={},
                finish=lambda **kwargs: closed.append(kwargs['exit_code']))
            trainer.accelerator.end_training = lambda: closed.append('accelerator')
            with patch.dict('sys.modules', {'wandb': tracker}):
                finish_deteriorated_training(trainer)
            self.assertEqual(tracker.summary['training_status'], 'stopped_deterioration')
            self.assertEqual(tracker.summary['health_stop_reason']['global_step'], 3)
            self.assertEqual(tracker.summary['health_best_checked_step'], 1)
            self.assertEqual(closed, [1, 'accelerator'])


class DeteriorationContractTests(unittest.TestCase):
    def test_enabled_policy_requires_recoverable_health_cadence(self):
        from scripts.train.shared.contracts import validate_sft_reliability
        valid = {'recovery': {'before_evaluation':True,'save_steps':100,'save_total_limit':2},
                 'health_check': dict(POLICY, every_n_steps=100,max_examples=32)}
        validate_sft_reliability(valid)
        for modification in ({'recovery':None},
                             {'recovery':dict(valid['recovery'],save_steps=150)},
                             {'health_check':dict(valid['health_check'],stop_on_deterioration={
                                 'loss_factor':1.,'loss_delta':.5,'consecutive_checks':2})}):
            with self.assertRaises(ValueError):
                validate_sft_reliability(dict(valid, **modification))


if __name__ == '__main__':
    unittest.main()
