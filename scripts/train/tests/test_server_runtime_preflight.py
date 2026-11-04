"""Reject missing server patches before creating a launch bundle or loading weights."""
import subprocess
import hashlib
import json
import sys
import unittest
import tempfile
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

from scripts.train.psc import launch


class ServerRuntimePreflightTests(unittest.TestCase):
    def test_subject_only_answer_run_requires_readonly_patch_check(self):
        run = {'answer_format': 'answer_only', 'subject_check': {'enabled': True}}
        with patch.object(launch.subprocess, 'run') as check:
            launch.verify_server_runtime(run)
        check.assert_called_once_with(
            [sys.executable, '-m', 'scripts.train.rl.rollout.patch'], check=True)

    def test_cot_requires_check_even_without_subject_task(self):
        with patch.object(launch.subprocess, 'run') as check:
            launch.verify_server_runtime({'answer_format': 'cot'})
        self.assertEqual(check.call_count, 1)

    def test_missing_patch_blocks_launch_without_automatic_mutation(self):
        error = subprocess.CalledProcessError(1, ['patch-check'])
        with patch.object(launch.subprocess, 'run', side_effect=error) as check:
            with self.assertRaises(subprocess.CalledProcessError):
                launch.verify_server_runtime({'answer_format': 'answer_only', 'subject_check': {'enabled': True}})
        self.assertNotIn('--apply', check.call_args.args[0])

    def test_prepare_does_not_publish_bundle_when_runtime_check_fails(self):
        run = {'model': 'qwen3.5-9b', 'recipe': 'sft', 'answer_format': 'answer_only',
               'subject_check': {'enabled': True}, 'env_file': 'unused',
               'execution': {'training_gpus': 8, 'vllm_gpu_sharing': True}}
        error = subprocess.CalledProcessError(1, ['patch-check'])
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary) / 'not-created'
            stack.enter_context(patch.object(sys, 'argv', ['launch', 'prepare', '--scheduler', 'pbs',
                '--allocated-nodes', '1', '--config', 'unused', '--job-root', str(root), '--master', 'node']))
            stack.enter_context(patch('scripts.train.shared.run_config.load_run', return_value=run))
            stack.enter_context(patch('scripts.train.shared.run_config.compile_run',
                return_value={'run': run, 'experiment': {}, 'stages': [{}]}))
            stack.enter_context(patch('scripts.train.run.preflight'))
            writer = stack.enter_context(patch('scripts.train.run.write_bundle'))
            stack.enter_context(patch('dotenv.dotenv_values', return_value={'WANDB_API_KEY': 'test'}))
            stack.enter_context(patch('scripts.train.sft.launch.resolve_training_args',
                return_value={'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 2}))
            stack.enter_context(patch('scripts.train.sft.launch.build_callbacks'))
            stack.enter_context(patch.object(launch.subprocess, 'run', side_effect=error))
            with self.assertRaises(subprocess.CalledProcessError):
                launch.main()
            self.assertFalse(root.exists())
            writer.assert_not_called()

    def test_existing_medical_only_runtime_is_unchanged(self):
        with patch.object(launch.subprocess, 'run') as check:
            launch.verify_server_runtime({'answer_format': 'answer_only'})
        check.assert_not_called()


class WeightProvenancePreflightTests(unittest.TestCase):
    def prepare_fixture(self, base, with_manifest):
        model = {'repository': 'google/gemma-4-31B-it', 'revision': 'pinned'}
        weights = base / 'weights' / model['repository'] / model['revision']
        weights.mkdir(parents=True)
        payload = weights / 'config.json'
        payload.write_bytes(b'{}')
        if with_manifest:
            (weights / 'cloud_source.json').write_text(json.dumps({**model, 'source': {}, 'files': {
                'config.json': {'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest(),
                                'mtime_ns': payload.stat().st_mtime_ns}}}))
        run = {'model': 'gemma4-31b-it', 'model_settings': model, 'recipe': 'sft',
               'answer_format': 'answer_only', 'env_file': 'unused',
               'paths': {'weights_root': str(base / 'weights')},
               'execution': {'training_gpus': 16, 'vllm_gpu_sharing': True}}
        root = base / 'run'
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', ['launch', 'prepare', '--scheduler', 'pbs',
                '--allocated-nodes', '2', '--config', 'unused', '--job-root', str(root), '--master', 'node']))
            stack.enter_context(patch('scripts.train.shared.run_config.load_run', return_value=run))
            stack.enter_context(patch('scripts.train.shared.run_config.compile_run',
                return_value={'run': run, 'experiment': {}, 'stages': [{}]}))
            stack.enter_context(patch('scripts.train.run.preflight'))
            writer = stack.enter_context(patch('scripts.train.run.write_bundle'))
            stack.enter_context(patch('dotenv.dotenv_values', return_value={'WANDB_API_KEY': 'test'}))
            stack.enter_context(patch('scripts.train.sft.launch.resolve_training_args',
                return_value={'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 1}))
            stack.enter_context(patch('scripts.train.sft.launch.build_callbacks'))
            if with_manifest:
                launch.main()
                writer.assert_called_once()
                self.assertTrue((root / 'plan.json').is_file())
            else:
                with self.assertRaises(FileNotFoundError):
                    launch.main()
                writer.assert_not_called()
                self.assertFalse(root.exists())

    def test_missing_manifest_blocks_prepare_before_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.prepare_fixture(Path(temporary), with_manifest=False)

    def test_valid_manifest_allows_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.prepare_fixture(Path(temporary), with_manifest=True)


if __name__ == '__main__':
    unittest.main()
