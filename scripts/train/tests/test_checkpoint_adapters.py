import unittest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


from scripts.train.shared.peft_compat import install_checkpoint_adapter_compatibility, validate_adapter_state


class CheckpointAdapterTests(unittest.TestCase):
    def setUp(self):
        install_checkpoint_adapter_compatibility()

    def test_completeness_guard_rejects_extra_and_wrong_shape(self):
        model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)),
                               LoraConfig(r=2, target_modules=['0']))
        weights = get_peft_model_state_dict(model)
        key = next(iter(weights))
        for invalid in ({}, {**weights, 'extra': torch.zeros(1)},
                        {**weights, key: torch.zeros(1)}):
            with self.subTest(keys=list(invalid)), self.assertRaises(ValueError):
                validate_adapter_state(model, invalid)

    def test_wrapped_adapter_save_and_restore_is_complete(self):
        model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)),
                               LoraConfig(r=2, target_modules=['0', '1']))
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'lora_' in name:
                    value.fill_(0.25)
        expected = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
        model.base_model.model[0] = checkpoint_wrapper(model.base_model.model[0])
        saved = get_peft_model_state_dict(model, state_dict=model.state_dict())
        self.assertEqual(set(saved), set(expected))
        self.assertEqual(validate_adapter_state(model, saved), 4)
        with self.assertRaisesRegex(ValueError, "Incomplete adapter"):
            validate_adapter_state(model, {next(iter(saved)): next(iter(saved.values()))})
        saved = {key: value.clone() for key, value in saved.items()}
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'lora_' in name:
                    value.zero_()
        result = set_peft_model_state_dict(model, saved)
        self.assertFalse(result.unexpected_keys)
        restored = get_peft_model_state_dict(model)
        for name in expected:
            torch.testing.assert_close(restored[name], expected[name])


if __name__ == '__main__':
    unittest.main()
