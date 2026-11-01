import os
import unittest
from unittest.mock import patch
from scripts.train.shared.vllm import weight_transfer_port

class TransferPortTests(unittest.TestCase):
    def test_default_preserves_existing_run(self):
        with patch.dict(os.environ,{},clear=True): self.assertEqual(weight_transfer_port(),51216)

    def test_explicit_per_run_port(self):
        with patch.dict(os.environ,{'TRIAGE_VLLM_GROUP_PORT':'51223'}): self.assertEqual(weight_transfer_port(),51223)

    def test_invalid_port_fails(self):
        for value in ('0','80','65536','bad'):
            with patch.dict(os.environ,{'TRIAGE_VLLM_GROUP_PORT':value}),self.assertRaises(ValueError):weight_transfer_port()

    def test_sync_passes_assigned_port_to_trl(self):
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import Mock
        from scripts.train.shared.vllm import SFTWeightSync
        fake=ModuleType('trl.generation.vllm_generation')
        fake.VLLMGeneration=Mock()
        weights=ModuleType('scripts.train.rl.rollout.weights')
        weights.install_policy_weight_export=Mock()
        trainer=SimpleNamespace(accelerator=Mock(),model_wrapped=object(),processing_class=object())
        with patch.dict(sys.modules,{'trl.generation.vllm_generation':fake,'scripts.train.rl.rollout.weights':weights}),patch.dict(os.environ,{'TRIAGE_VLLM_GROUP_PORT':'51224'}):
            SFTWeightSync(trainer,'http://127.0.0.1:8021').sync_weights()
            self.assertEqual(fake.VLLMGeneration.call_args.kwargs['group_port'],51224)
            self.assertEqual(fake.VLLMGeneration.call_args.kwargs['server_base_url'],'http://127.0.0.1:8021')
