"""Recovery checkpoints must preserve progress without selecting a best model."""
import copy
from datetime import timedelta
import importlib
import json
from pathlib import Path
import random
import os
import socket
import tempfile
import unittest

import numpy as np
import torch
from transformers import Trainer, TrainingArguments
import torch.distributed as dist
import torch.multiprocessing as mp


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, labels=None):
        logits = self.weight * input_ids.float()
        return {'loss': ((logits - labels.float()) ** 2).mean(), 'logits': logits}


def distributed_recovery_worker(rank, port, root, fail):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=5))
    from scripts.train.sft.recovery import RecoveryCheckpointMixin

    class NativeTrainer(Trainer):
        def save_model(self, output_dir=None, _internal_call=False):
            super().save_model(output_dir, _internal_call=_internal_call)
            if fail == 'early_model' and rank == 0:
                raise OSError('rank zero model write failure')

        def _save_optimizer_and_scheduler(self, output_dir):
            if fail == 'early_model':
                Path(root, f'optimizer-entered-{rank}').touch()
                dist.all_reduce(torch.ones(1))
            super()._save_optimizer_and_scheduler(output_dir)

        def _save_rng_state(self, output_dir):
            super()._save_rng_state(output_dir)
            if fail is True and rank == 1:
                raise OSError('rank one disk failure')

    class RecoveryTrainer(RecoveryCheckpointMixin, NativeTrainer):
        pass

    try:
        trainer = RecoveryTrainer(model=TinyModel(), args=TrainingArguments(
            output_dir=str(Path(root, 'checkpoints')), use_cpu=True,
            max_steps=1, per_device_train_batch_size=1, report_to='none',
            disable_tqdm=True, save_strategy='no', ddp_find_unused_parameters=False, ddp_timeout=5),
            train_dataset=[{'input_ids': torch.ones(1), 'labels': torch.zeros(1)}] * 4)
        trainer.configure_recovery({'before_evaluation': True, 'save_steps': 1, 'save_total_limit': 2})
        arguments = trainer.args
        error = None
        try:
            trainer.train()
        except (OSError, RuntimeError) as exc:
            error = str(exc)
        Path(root, f'rank-{rank}.json').write_text(json.dumps({
            'error': error, 'args_restored': trainer.args is arguments,
            'saving': trainer._recovery_saving}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class RecoveryTests(unittest.TestCase):
    def recovery_class(self, base=Trainer):
        try:
            module = importlib.import_module('scripts.train.sft.recovery')
        except ModuleNotFoundError:
            self.fail('Recovery checkpoint implementation is missing')
        return type('RecoveryTrainer', (module.RecoveryCheckpointMixin, base), {})

    def trainer(self, root, *, base=Trainer, policy=True, **kwargs):
        cls = self.recovery_class(base)
        arguments = dict(output_dir=str(Path(root, 'checkpoints')), use_cpu=True,
                         report_to='none', save_strategy='no', max_steps=3,
                         per_device_train_batch_size=1, disable_tqdm=True,
                         learning_rate=.01, seed=43)
        arguments.update(kwargs)
        trainer = cls(model=TinyModel(), args=TrainingArguments(**arguments),
                      train_dataset=[{'input_ids': torch.ones(1), 'labels': torch.zeros(1)}] * 6,
                      eval_dataset=[{'input_ids': torch.ones(1), 'labels': torch.zeros(1)}])
        trainer.configure_recovery({'before_evaluation': True, 'save_steps': 2,
                                    'save_total_limit': 2} if policy else None)
        return trainer

    def test_periodic_and_eval_boundary_save_before_evaluation_failure(self):
        class FailedEvaluation(Trainer):
            def evaluate(self, *args, **kwargs):
                raise RuntimeError('generation failed')
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, base=FailedEvaluation, eval_strategy='steps', eval_steps=3)
            with self.assertRaisesRegex(RuntimeError, 'generation failed'):
                trainer.train()
            recovery = Path(root, 'recovery')
            self.assertEqual(sorted(p.name for p in recovery.glob('checkpoint-*')),
                             ['checkpoint-2', 'checkpoint-3'])
            for step in (2, 3):
                checkpoint = recovery / f'checkpoint-{step}'
                for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt',
                             'rng_state.pth', 'trainer_state.json', 'recovery_complete.json'):
                    self.assertTrue((checkpoint / name).is_file(), name)
                self.assertEqual(json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'], step)

    def test_disabled_policy_preserves_no_save_behavior(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, policy=False)
            trainer.train()
            self.assertFalse(Path(root, 'recovery').exists())

    def test_optimizer_rng_and_model_resume_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root)
            trainer.train()
            checkpoint = Path(root, 'recovery', 'checkpoint-2')
            saved_rng = torch.load(checkpoint / 'rng_state.pth', weights_only=False)
            random.setstate(saved_rng['python'])
            np.random.set_state(saved_rng['numpy'])
            torch.random.set_rng_state(saved_rng['cpu'])
            expected_random = (random.random(), np.random.random(), torch.rand(1))
            random.seed(17); np.random.seed(17); torch.manual_seed(17)
            trainer._load_rng_state(str(checkpoint))
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.random(), expected_random[1])
            torch.testing.assert_close(torch.rand(1), expected_random[2])
            resumed = self.trainer(root, policy=False)
            resumed.train(resume_from_checkpoint=str(checkpoint))
            torch.testing.assert_close(resumed.model.weight, trainer.model.weight, rtol=0, atol=0)
            self.assertEqual(resumed.state.global_step, 3)
            self.assertEqual(resumed.lr_scheduler.state_dict(), trainer.lr_scheduler.state_dict())
            for key, value in trainer.optimizer.state_dict()['state'][0].items():
                torch.testing.assert_close(resumed.optimizer.state_dict()['state'][0][key], value, rtol=0, atol=0)

    def test_rotation_does_not_touch_regular_best_or_incomplete_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            regular = Path(root, 'checkpoints', 'checkpoint-1')
            regular.mkdir(parents=True)
            (regular / 'sentinel').write_text('selected model')
            incomplete = Path(root, 'recovery', '.incomplete-99')
            incomplete.mkdir(parents=True)
            trainer = self.trainer(root, max_steps=8)
            trainer.state.best_metric = .75
            trainer.state.best_global_step = 1
            trainer.state.best_model_checkpoint = str(regular)
            trainer.train()
            self.assertEqual(sorted(p.name for p in Path(root, 'recovery').glob('checkpoint-*')),
                             ['checkpoint-6', 'checkpoint-8'])
            self.assertEqual((regular / 'sentinel').read_text(), 'selected model')
            self.assertTrue(incomplete.exists())

    def test_save_keeps_best_metadata_flags_and_callback_state_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, save_strategy='steps', save_steps=2,
                                   save_total_limit=1, save_only_model=True)
            trainer.create_optimizer_and_scheduler(num_training_steps=3)
            trainer.state.global_step = 2
            trainer.state.best_metric = .75
            trainer.state.best_global_step = 2
            trainer.state.best_model_checkpoint = str(Path(root, 'checkpoints', 'checkpoint-2'))
            trainer.is_in_train = True
            trainer.control.should_save = True
            trainer.control.should_evaluate = True
            state, control, arguments = trainer.state, trainer.control, trainer.args
            state_before = copy.deepcopy(vars(state))
            control_before = copy.deepcopy(vars(control))
            trainer.save_recovery_before_evaluation()
            checkpoint = Path(root, 'recovery', 'checkpoint-2')
            self.assertTrue((checkpoint / 'optimizer.pt').exists())
            self.assertIs(trainer.state, state)
            self.assertIs(trainer.control, control)
            self.assertIs(trainer.args, arguments)
            self.assertEqual(vars(state), state_before)
            self.assertEqual(vars(control), control_before)
            persisted = json.loads((checkpoint / 'trainer_state.json').read_text())
            self.assertEqual(persisted['best_model_checkpoint'], state.best_model_checkpoint)
            self.assertEqual(persisted['best_metric'], .75)
            marker = checkpoint / 'recovery_complete.json'
            before = marker.stat().st_mtime_ns
            trainer.save_recovery_before_evaluation()
            self.assertEqual(marker.stat().st_mtime_ns, before)

    def test_completed_checkpoint_is_reused_by_a_new_trainer_at_the_same_step(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root)
            trainer.train()
            marker = Path(root, 'recovery', 'checkpoint-2', 'recovery_complete.json')
            before = marker.stat().st_mtime_ns
            restored = self.trainer(root)
            restored.create_optimizer_and_scheduler(num_training_steps=3)
            restored.state.global_step = 2
            restored.is_in_train = True
            restored.save_recovery_before_evaluation()
            self.assertEqual(marker.stat().st_mtime_ns, before)

    def test_all_ranks_save_rng_and_share_a_rank_local_write_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as root:
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                mp.spawn(distributed_recovery_worker, args=(port, root, fail), nprocs=2, join=True)
                for rank in (0, 1):
                    result = json.loads(Path(root, f'rank-{rank}.json').read_text())
                    self.assertTrue(result['args_restored'])
                    self.assertFalse(result['saving'])
                    if fail:
                        self.assertIn('rank one disk failure', result['error'])
                    else:
                        self.assertIsNone(result['error'])
                        self.assertTrue(Path(root, 'recovery', 'checkpoint-1', f'rng_state_{rank}.pth').is_file())
                marker = Path(root, 'recovery', 'checkpoint-1', 'recovery_complete.json')
                self.assertEqual(marker.exists(), not fail)
                if not fail:
                    self.assertEqual(json.loads(marker.read_text()),
                                     {'schema_version': 1, 'global_step': 1, 'completed': True})

    def test_rank_zero_model_io_error_prevents_later_collective_on_every_rank(self):
        with tempfile.TemporaryDirectory() as root:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            mp.spawn(distributed_recovery_worker, args=(port, root, 'early_model'), nprocs=2, join=True)
            for rank in (0, 1):
                result = json.loads(Path(root, f'rank-{rank}.json').read_text())
                self.assertIn('rank zero model write failure', result['error'])
                self.assertTrue(result['args_restored'])
                self.assertFalse(result['saving'])
                self.assertFalse(Path(root, f'optimizer-entered-{rank}').exists())
            self.assertFalse(list(Path(root, 'recovery').rglob('recovery_complete.json')))

    def test_save_failure_preserves_live_state_and_does_not_publish_partial_checkpoint(self):
        class DiskFailure(Trainer):
            def _save_optimizer_and_scheduler(self, output_dir):
                super()._save_optimizer_and_scheduler(output_dir)
                raise OSError('disk full')
        with tempfile.TemporaryDirectory() as root:
            trainer = self.trainer(root, base=DiskFailure, save_only_model=True)
            original_args, original_control = trainer.args, trainer.control
            with self.assertRaisesRegex(OSError, 'disk full'):
                trainer.train()
            self.assertIs(trainer.args, original_args)
            self.assertIs(trainer.control, original_control)
            self.assertTrue(trainer.args.save_only_model)
            self.assertEqual(trainer.args.output_dir, str(Path(root, 'checkpoints')))
            self.assertEqual(trainer.state.global_step, 2)
            self.assertFalse(Path(root, 'recovery', 'checkpoint-2').exists())
            self.assertFalse(list(Path(root, 'recovery').rglob('recovery_complete.json')))


if __name__ == '__main__':
    unittest.main()
