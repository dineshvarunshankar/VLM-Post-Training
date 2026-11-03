"""PSC CoT must retain 16 training ranks and sequential shared evaluation."""
import importlib
from pathlib import Path
import unittest


class PSCLaunchTests(unittest.TestCase):
    def module(self):
        try:
            module = importlib.import_module('scripts.train.psc.launch')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'Reusable PSC CoT launcher is not implemented')
        return module

    def test_two_nodes_keep_all_ranks_and_gemma_runtime(self):
        module = self.module()
        for model in ('qwen3.8-27b', 'gemma4-31b-it'):
            plan = {'run': {'model': model, 'answer_format': 'cot',
                           'execution': {'training_gpus': 16, 'training_devices': None,
                                         'vllm_gpu_sharing': True}}}
            command = module.training_command(plan, ['torchrun', 'train_sft.py', '--config', 'bundle.yaml'],
                script='train_sft.py', master='node0', rank='1', job_id='123', python='/env/python')
            self.assertEqual(command[command.index('--nnodes') + 1], '2')
            self.assertEqual(command[command.index('--nproc-per-node') + 1], '8')
            self.assertEqual(command[-3:], ['train_sft.py', '--config', 'bundle.yaml'])
            self.assertEqual(any('gemma_dynamic_flex_entry.py' in x for x in command), model.startswith('gemma'))
            flags = module.evaluation_arguments(plan['run'])
            self.assertEqual(flags, ['--reasoning-parser', 'gemma4' if model.startswith('gemma') else 'qwen3'])

    def test_mismatched_layout_fails_instead_of_overriding_config(self):
        module = self.module()
        for ranks, sharing in ((8, True), (16, False)):
            run = {'model': 'qwen3.8-27b', 'answer_format': 'cot',
                   'execution': {'training_gpus': ranks, 'vllm_gpu_sharing': sharing, 'training_devices': None}}
            with self.assertRaises(ValueError):
                module.validate_layout(run)


if __name__ == '__main__':
    unittest.main()
