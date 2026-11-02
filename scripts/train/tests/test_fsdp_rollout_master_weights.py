"""Export original FSDP2 master weights after no-grad mixed-precision evaluation."""
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from scripts.train.rl.rollout.weights import iter_policy_weights


class FSDPRolloutMasterWeightsTest(unittest.TestCase):
    def test_export_restores_master_adapter_precision_after_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            init_process_group('gloo', init_method=f'file://{directory}/rdzv', rank=0, world_size=1)
            try:
                model = get_peft_model(torch.nn.Sequential(
                    torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)),
                    LoraConfig(r=2, lora_alpha=2, target_modules=['0']))
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if 'lora_' in name:
                            parameter.fill_(0.1234567)
                expected = {name: value.clone() for name, value in iter_policy_weights(model)}
                masters = {name: value.detach().clone() for name, value in model.named_parameters()}
                fully_shard(model, mesh=init_device_mesh('cpu', (1,)),
                            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
                            reshard_after_forward=False)
                with torch.no_grad():
                    model(torch.ones(2, 4, dtype=torch.bfloat16))
                adapters = [p for n, p in model.named_parameters() if 'lora_' in n]
                self.assertTrue(all(p.dtype == torch.bfloat16 and not isinstance(p, DTensor) for p in adapters))
                actual = dict(iter_policy_weights(model))
                self.assertEqual(actual.keys(), expected.keys())
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    self.assertIsInstance(parameter, DTensor)
                    torch.testing.assert_close(parameter.full_tensor(), masters[name], rtol=0, atol=0)
                model(torch.ones(2, 4, dtype=torch.bfloat16)).float().sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad.full_tensor()).all()
                                    for p in model.parameters() if p.requires_grad))
            finally:
                destroy_process_group()

if __name__ == '__main__':
    unittest.main()
