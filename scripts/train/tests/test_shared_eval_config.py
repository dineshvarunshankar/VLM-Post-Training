import unittest
from pathlib import Path
from scripts.train.shared.run_config import compile_run, load_run
from scripts.train.run import stage_command

class SharedEvalConfigTests(unittest.TestCase):
    def setUp(self):
        self.run = load_run('scripts/train/configs/runs/qwen9b_sft_stabilized_medical.yaml')
    def test_dedicated_default_has_no_sharing_flag(self):
        self.run['execution'].pop('vllm_gpu_sharing', None)
        plan = compile_run(self.run)
        self.assertNotIn('--vllm-gpu-sharing', stage_command(plan, plan['stages'][0], Path('/tmp/bundle')))
    def test_sharing_is_explicit_and_reaches_sft(self):
        self.run['execution']['vllm_gpu_sharing'] = True
        plan = compile_run(self.run)
        self.assertIn('--vllm-gpu-sharing', stage_command(plan, plan['stages'][0], Path('/tmp/bundle')))
    def test_sharing_rejects_non_boolean_values(self):
        for value in ('true', 1, None):
            with self.subTest(value=value):
                self.run['execution']['vllm_gpu_sharing'] = value
                with self.assertRaisesRegex(ValueError, 'vllm_gpu_sharing'):
                    compile_run(self.run)

if __name__ == '__main__':
    unittest.main()
