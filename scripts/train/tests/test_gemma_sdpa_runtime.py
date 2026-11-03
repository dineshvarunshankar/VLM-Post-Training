"""Backend selection and dropout replay must remain independent."""
import unittest
from pathlib import Path
import torch
import yaml
from scripts.train.psc.launch import training_command
from scripts.train.shared.run_config import load_run, compile_run

ROOT = Path(__file__).resolve().parents[3]

class GemmaRuntimeTests(unittest.TestCase):
    def test_sdpa_does_not_install_flex_entry(self):
        plan = {'run': {'model': 'gemma4-31b-it', 'execution': {
            'training_gpus':16, 'training_devices':None, 'vllm_gpu_sharing':True}},
            'experiment': {'attention':'sdpa'}}
        command = training_command(plan, ['python','train.py'], script='train.py',
            master='node0',rank='0',job_id='test',python='python')
        self.assertFalse(any('flex_entry' in x for x in command), command)
        plan['experiment']['attention'] = 'flex_attention'
        command = training_command(plan, ['python','train.py'], script='train.py',
            master='node0',rank='0',job_id='test',python='python')
        self.assertTrue(any('flex_entry' in x for x in command))
        # The PSC large-model entrypoint must not silently adopt 12B.
        plan['run']['model'] = 'gemma4-12b-it'
        with self.assertRaisesRegex(ValueError, 'Qwen 27B and Gemma 31B only'):
            training_command(plan, ['python','train.py'], script='train.py',
                master='node0',rank='0',job_id='test',python='python')

    def test_future_gemma_configs_retain_flex_and_approved_learning_rate(self):
        paths = list((ROOT/'scripts/train/configs/runs').glob('gemma*sft*.yaml'))
        paths += list((ROOT/'scripts/train/configs/cot').glob('gemma*.yaml'))
        self.assertEqual(len(paths),5)
        for path in paths:
            with self.subTest(path=path.name):
                raw=yaml.safe_load(path.read_text())
                for key in ('media_root','split_dir'):
                    if raw['paths'][key] is None:raw['paths'][key]='/explicit/test-release'
                plan=compile_run(raw)
                self.assertEqual(plan['experiment']['attention'],'flex_attention')
                self.assertEqual(plan['experiment']['training']['learning_rate'],2e-5)

    def test_sdpa_fsdp_config_preserves_dropout_replay(self):
        from scripts.train.sft.launch import build_training_config
        from torch.distributed.algorithms._checkpoint import checkpoint_wrapper as module
        build_training_config(Path('/tmp/unused-gemma-runtime'), {
            'fsdp':True,'fsdp_config':{'version':2,'activation_checkpointing':True},
            'use_cpu':True})
        # Accelerate passes False; the shared compatibility fix must override it.
        linear=torch.nn.Sequential(torch.nn.Dropout(.5),torch.nn.Linear(8,1,bias=False))
        wrapped=module.checkpoint_wrapper(linear,preserve_rng_state=False)
        x=torch.ones(4,8,requires_grad=True)
        torch.manual_seed(11)
        linear(x).sum().backward()
        expected=linear[1].weight.grad.clone()
        linear.zero_grad();x.grad=None
        torch.manual_seed(11)
        wrapped(x).sum().backward()
        torch.testing.assert_close(linear[1].weight.grad,expected,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
