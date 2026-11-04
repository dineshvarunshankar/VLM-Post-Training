"""Retain the best checked recovery checkpoint and its exact health history."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Trainer, TrainingArguments

from scripts.train.tests import test_sft_recovery as recovery_tests
from scripts.train.tests.test_sft_recovery import TinyModel


def health_state(step=None, best=None, value=None):
    return {'schema_version': 1, 'policy': {'every_n_steps': 2, 'max_examples': 32},
            'subset_sha256': 'fixed-test-subset', 'subset_indices': [0],
            'last_checked_step': step, 'best_step': best, 'best_value_loss': value,
            'bad_checks': 0, 'stopped': False, 'stop_reason': None}


def ready_trainer(root):
    trainer = recovery_tests.RecoveryTests().trainer(root)
    trainer.create_optimizer_and_scheduler(num_training_steps=10)
    trainer.is_in_train = True
    trainer._health_state = health_state()
    return trainer


def save_step(trainer, step):
    trainer.state.global_step = step
    trainer.save_recovery_before_evaluation()


def publish_failure_worker(rank, port, root):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(seconds=5))
    from scripts.train.sft.recovery import RecoveryCheckpointMixin
    from scripts.train.shared.data import atomic_json

    class RecoveryTrainer(RecoveryCheckpointMixin, Trainer):
        pass

    try:
        trainer = RecoveryTrainer(model=TinyModel(), args=TrainingArguments(
            output_dir=str(Path(root, 'checkpoints')), use_cpu=True, max_steps=1,
            per_device_train_batch_size=1, report_to='none', disable_tqdm=True,
            save_strategy='no', ddp_find_unused_parameters=False),
            train_dataset=[{'input_ids': torch.ones(1), 'labels': torch.zeros(1)}] * 4)
        trainer.configure_recovery({'before_evaluation': True, 'save_steps': 1, 'save_total_limit': 2})
        trainer.train()
        state = health_state(1, 1, .2)
        error = None

        def fail_root_write(path, value):
            if Path(path) == Path(root, 'recovery', 'health_state.json'):
                raise OSError('pin publication failed')
            atomic_json(path, value)

        try:
            with patch('scripts.train.shared.data.atomic_json', side_effect=fail_root_write):
                trainer.publish_health_state(state, pin_best=True)
        except (OSError, RuntimeError) as exc:
            error = str(exc)
        Path(root, f'publication-rank-{rank}.json').write_text(json.dumps({'error': error}))
    finally:
        dist.destroy_process_group()


class RecoveryHealthTests(unittest.TestCase):
    def require_interface(self, trainer):
        self.assertTrue(hasattr(trainer, 'publish_health_state'), 'Recovery health-state publication is missing')
        self.assertTrue(hasattr(trainer, 'restore_health_recovery_state'), 'Recovery health-state restore is missing')

    def test_pinned_best_survives_rotation_before_failing_health_then_new_best_replaces_it(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = ready_trainer(root)
            self.require_interface(trainer)
            save_step(trainer, 2)
            trainer._health_state = health_state(2, 2, .2)
            trainer.publish_health_state(trainer._health_state, pin_best=True)
            for step in (4, 6, 8):
                save_step(trainer, step)
            directory = Path(root, 'recovery')
            self.assertEqual(sorted(p.name for p in directory.glob('checkpoint-*')),
                             ['checkpoint-2', 'checkpoint-6', 'checkpoint-8'])
            self.assertEqual(json.loads((directory / 'checkpoint-8' / 'health_state.json').read_text()),
                             health_state(2, 2, .2))
            improved = health_state(8, 8, .1)
            trainer._health_state = improved
            trainer.publish_health_state(improved, pin_best=True)
            self.assertEqual(sorted(p.name for p in directory.glob('checkpoint-*')),
                             ['checkpoint-6', 'checkpoint-8'])
            self.assertEqual(json.loads((directory / 'health_state.json').read_text()), improved)
            self.assertEqual(json.loads((directory / 'checkpoint-8' / 'health_state.json').read_text()), improved)

    def test_initial_unchecked_history_is_saved_with_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = ready_trainer(root)
            self.require_interface(trainer)
            save_step(trainer, 2)
            saved = Path(root, 'recovery', 'checkpoint-2', 'health_state.json')
            self.assertEqual(json.loads(saved.read_text()), health_state())

    def test_resume_republishes_checkpoint_local_history_instead_of_future_root_history(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = ready_trainer(root)
            self.require_interface(trainer)
            save_step(trainer, 2)
            earlier = health_state(2, 2, .2)
            trainer._health_state = earlier
            trainer.publish_health_state(earlier, pin_best=True)
            save_step(trainer, 4)
            trainer._health_state = health_state(4, 4, .1)
            trainer.publish_health_state(trainer._health_state, pin_best=True)
            resumed = ready_trainer(root)
            resumed._health_state = json.loads(Path(root, 'recovery', 'checkpoint-2', 'health_state.json').read_text())
            resumed.restore_health_recovery_state(resumed._health_state, resume_step=2)
            self.assertEqual(json.loads(Path(root, 'recovery', 'health_state.json').read_text()), earlier)
            for step in (6, 8):
                save_step(resumed, step)
            self.assertTrue(Path(root, 'recovery', 'checkpoint-2').is_dir())
            self.assertFalse(Path(root, 'recovery', 'checkpoint-4').exists())
            self.assertEqual(json.loads(Path(root, 'recovery', 'checkpoint-8', 'health_state.json').read_text()), earlier)

    def test_rewind_preserves_future_artifacts_and_saves_current_branch_weights(self):
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as root:
            original = ready_trainer(root)
            save_step(original, 2)
            earlier = health_state(2, 2, .2)
            original._health_state = earlier
            original.publish_health_state(earlier, pin_best=True)
            for step in (4, 6, 8):
                save_step(original, step)
            prior_model = Path(root, 'recovery', 'checkpoint-6', 'model.safetensors').read_bytes()
            resumed = ready_trainer(root)
            resumed._health_state = copy.deepcopy(earlier)
            resumed.restore_health_recovery_state(earlier, resume_step=2)
            with torch.no_grad():
                resumed.model.weight.fill_(7.)
            save_step(resumed, 4)
            self.assertTrue(Path(root, 'recovery', 'checkpoint-4').is_dir())
            resumed.publish_health_state(health_state(4, 2, .2))
            with torch.no_grad():
                resumed.model.weight.fill_(9.)
            save_step(resumed, 6)
            fresh = load_file(str(Path(root, 'recovery', 'checkpoint-6', 'model.safetensors')))
            torch.testing.assert_close(fresh['weight'], torch.tensor([9.]))
            archived = list(Path(root, 'recovery').glob('.superseded-*/checkpoint-6/model.safetensors'))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), prior_model)
            self.assertTrue(Path(root, 'recovery', 'checkpoint-2').is_dir())

    def test_publication_failure_preserves_previous_pin(self):
        from scripts.train.shared.data import atomic_json
        with tempfile.TemporaryDirectory() as root:
            trainer = ready_trainer(root)
            self.require_interface(trainer)
            save_step(trainer, 2)
            previous = health_state(2, 2, .2)
            trainer._health_state = previous
            trainer.publish_health_state(previous, pin_best=True)
            save_step(trainer, 4)
            improved = health_state(4, 4, .1)
            root_state = Path(root, 'recovery', 'health_state.json')

            def fail_root_write(path, value):
                if Path(path) == root_state:
                    raise OSError('pin publication failed')
                atomic_json(path, value)

            with patch('scripts.train.shared.data.atomic_json', side_effect=fail_root_write):
                with self.assertRaisesRegex(OSError, 'pin publication failed'):
                    trainer.publish_health_state(improved, pin_best=True)
            self.assertEqual(json.loads(root_state.read_text()), previous)
            self.assertTrue(Path(root, 'recovery', 'checkpoint-2').is_dir())
            self.assertEqual(json.loads(Path(root, 'recovery', 'checkpoint-4', 'health_state.json').read_text()), improved)

    def test_pin_requires_a_completed_checkpoint_at_the_actual_checked_step(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = ready_trainer(root)
            self.require_interface(trainer)
            save_step(trainer, 2)
            with self.assertRaises(ValueError):
                trainer.publish_health_state(health_state(3, 3, .1), pin_best=True)
            trainer_state = Path(root, 'recovery', 'checkpoint-2', 'trainer_state.json')
            values = json.loads(trainer_state.read_text())
            values['global_step'] = 1
            trainer_state.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                trainer.publish_health_state(health_state(2, 2, .1), pin_best=True)
            self.assertFalse(Path(root, 'recovery', 'health_state.json').exists())

    def test_missing_pinned_checkpoint_blocks_restore(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = ready_trainer(root)
            self.require_interface(trainer)
            with self.assertRaises(ValueError):
                trainer.restore_health_recovery_state(health_state(2, 2, .1), resume_step=2)
            self.assertFalse(Path(root, 'recovery', 'health_state.json').exists())

    def test_all_ranks_receive_health_metadata_write_failure(self):
        with tempfile.TemporaryDirectory() as root:
            self.require_interface(ready_trainer(root))
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            mp.spawn(publish_failure_worker, args=(port, root), nprocs=2, join=True)
            for rank in (0, 1):
                result = json.loads(Path(root, f'publication-rank-{rank}.json').read_text())
                self.assertIn('pin publication failed', result['error'])
            self.assertFalse(Path(root, 'recovery', 'health_state.json').exists())


if __name__ == '__main__':
    unittest.main()
