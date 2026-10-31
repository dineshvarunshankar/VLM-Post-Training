"""Tracking must not make another rank reject a fresh run before initialization."""
from contextlib import ExitStack
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.train.shared.runtime import validate_runtime
from scripts.train.shared.tracking import configure_tracking, add_tracking_metadata


class TrackingBootstrapTest(unittest.TestCase):
    def test_delayed_rank_accepts_same_fresh_run(self):
        # Exercise the real filesystem and bootstrap functions. Only CUDA hardware
        # checks are replaced; no process group or model is needed for this order.
        for mode in ('offline', 'online', 'disabled'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {
                    'WANDB_MODE': mode, 'WORLD_SIZE': '2', 'RANK': '0', 'LOCAL_RANK': '0',
                }))
                for name, result in (('is_available', True), ('device_count', 2),
                                     ('set_device', None), ('is_bf16_supported', True)):
                    stack.enter_context(patch('torch.cuda.' + name, return_value=result))
                output = Path(temporary) / 'run'
                args = SimpleNamespace(resume=None, run_name='bootstrap-race',
                                       run_date='2026-10-31', model='qwen3.8-9b',
                                       output_dir=output, output_root=None)
                profile = {'repository': 'Qwen/Qwen3.5-9B'}
                self.assertEqual(validate_runtime(args, profile), output)
                configure_tracking(args, 'sft_only', output, repository=profile['repository'])
                os.environ.update(RANK='1', LOCAL_RANK='1')
                try:
                    actual = validate_runtime(args, profile)
                except ValueError as error:
                    self.fail(f'A delayed rank rejected the fresh run: {error}')
                self.assertEqual(actual, output)

    def test_tracking_directory_is_deferred_until_main_trainer_exists(self):
        for mode in ('offline', 'online', 'disabled'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'WANDB_MODE': mode}):
                output = Path(temporary) / 'run'
                args = SimpleNamespace(resume=None, run_name='bootstrap-race',
                                       run_date='2026-10-31', model='qwen3.8-9b')
                settings = configure_tracking(args, 'sft_only', output)
                self.assertFalse(output.exists())
                report_to = [] if mode == 'disabled' else ['wandb']
                callbacks = []
                trainer = SimpleNamespace(args=SimpleNamespace(report_to=report_to),
                                          is_world_process_zero=lambda: False,
                                          add_callback=callbacks.append)
                add_tracking_metadata(trainer, output)
                self.assertFalse(output.exists())
                trainer.is_world_process_zero = lambda: True
                add_tracking_metadata(trainer, output)
                if mode == 'disabled':
                    self.assertFalse(output.exists())
                    self.assertEqual(settings['report_to'], 'none')
                else:
                    self.assertTrue((output / 'tracking').is_dir())
                    self.assertEqual(os.environ['WANDB_DIR'], str(output / 'tracking'))

    def test_resume_retains_existing_artifacts_and_tracking_identity(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'WANDB_MODE': 'offline', 'LOCAL_RANK': '0'}))
            for name, result in (('is_available', True), ('device_count', 1),
                                 ('set_device', None), ('is_bf16_supported', True)):
                stack.enter_context(patch('torch.cuda.' + name, return_value=result))
            output = Path(temporary) / 'run'
            checkpoint = output / 'checkpoints' / 'checkpoint-1'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'trainer_state.json').write_text('{}')
            metadata = {'run_location': {'date': '2026-10-31', 'name': 'original-name',
                                        'recipe': 'sft_only', 'answer_format': 'answer_only'}}
            (output / 'run.json').write_text(json.dumps(metadata))
            args = SimpleNamespace(resume=checkpoint, run_name='ignored-name',
                                   run_date='2026-10-31', model='qwen3.8-9b',
                                   output_dir=output, output_root=None)
            self.assertEqual(validate_runtime(args, {'repository': 'Qwen/Qwen3.5-9B'}), output)
            configure_tracking(args, 'sft_only', output)
            self.assertEqual(args.run_name, 'original-name')
            self.assertEqual(os.environ['WANDB_RESUME'], 'allow')
            self.assertFalse((output / 'tracking').exists())
            trainer = SimpleNamespace(args=SimpleNamespace(report_to=['wandb']),
                                      is_world_process_zero=lambda: True, add_callback=lambda _: None)
            add_tracking_metadata(trainer, output)
            self.assertTrue((output / 'tracking').is_dir())
            self.assertEqual(json.loads((output / 'run.json').read_text()), metadata)

    def test_fresh_run_still_rejects_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'existing-result.json').write_text('{}')
            args = SimpleNamespace(resume=None, run_name='bootstrap-race',
                                   run_date='2026-10-31', model='qwen3.8-9b',
                                   output_dir=output, output_root=None)
            with self.assertRaisesRegex(ValueError, 'Output directory is nonempty'):
                validate_runtime(args, {'repository': 'Qwen/Qwen3.5-9B'})


if __name__ == '__main__':
    unittest.main()
