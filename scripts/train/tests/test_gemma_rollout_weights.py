"""Runtime Gemma rollout names must match the original checkpoint layout."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from scripts.train.tests.test_cpu_attention_export import tiny_qwen
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper,
)
from scripts.train.tests.test_video_frame_compat import tiny_gemma
from scripts.train.shared.modeling import load_lora_model
from scripts.train.rl.rollout.weights import iter_policy_weights


class GemmaRolloutWeightsTest(unittest.TestCase):
    def test_real_lora_export_matches_saved_checkpoint_for_both_families(self):
        for unified in (True, False):
            with self.subTest(unified=unified), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = tiny_gemma(unified)
                source.save_pretrained(root / 'base')
                model, _ = load_lora_model(
                    {'local_path': str(root / 'base')}, source.config,
                    {'attention': 'sdpa', 'lora': {'r': 2, 'lora_alpha': 4, 'lora_dropout': 0.0}},
                )
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if 'lora_' in name:
                            parameter.fill_(0.125)
                exported = dict(iter_policy_weights(model))
                self.assertEqual(len(exported), len(list(iter_policy_weights(model))))
                # Exercise the actual activation-checkpoint wrapper name stripping.
                apply_activation_checkpointing(
                    model, checkpoint_wrapper_fn=checkpoint_wrapper,
                    check_fn=lambda module: type(module).__name__ in model.get_base_model()._no_split_modules,
                )
                wrapped = dict(iter_policy_weights(model))
                self.assertEqual(exported.keys(), wrapped.keys())
                for name in exported:
                    torch.testing.assert_close(exported[name], wrapped[name], rtol=0, atol=0)
                # The exporter does not mutate the frozen base tensors.
                for name, parameter in model.named_parameters():
                    canonical = name.replace('_checkpoint_wrapped_module.', '')
                    if 'lora_' not in canonical:
                        torch.testing.assert_close(parameter, before[canonical], rtol=0, atol=0)
                merged = model.merge_and_unload()
                merged.save_pretrained(root / 'merged')
                saved = {}
                for shard in (root / 'merged').glob('*.safetensors'):
                    saved.update(load_file(shard))
                # Serving already loads immutable buffers from the same base.
                # This stream transfers parameters, including every trained tensor.
                buffers = {name.replace('_checkpoint_wrapped_module.', '')
                           for name, _ in merged.named_buffers()}
                saved_buffers = set(saved) & buffers
                self.assertEqual(saved_buffers, {
                    f'model.language_model.layers.{i}.layer_scalar' for i in range(2)
                })
                for name in saved_buffers:
                    self.assertTrue(torch.equal(saved[name], torch.ones_like(saved[name])))
                expected = set(saved) - buffers
                actual = set(exported)
                self.assertEqual(actual - expected, set())
                self.assertEqual(expected - actual, set())
                for name, tensor in exported.items():
                    torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
                if unified:
                    self.assertIn('model.vision_embedder.pos_embedding', exported)
                    self.assertIn('model.vision_embedder.patch_dense.weight', exported)
                    self.assertIn('model.embed_vision.embedding_projection.weight', exported)
                    for checkpoint_name, runtime_name in (
                        ('model.vision_embedder.patch_dense.weight', 'model.embed_vision.patch_dense.base_layer.weight'),
                        ('model.embed_vision.embedding_projection.weight', 'model.embed_vision.multimodal_embedder.embedding_projection.base_layer.weight'),
                    ):
                        self.assertFalse(torch.equal(exported[checkpoint_name], before['base_model.model.' + runtime_name]))


    def test_real_qwen_export_names_are_unchanged(self):
        model = get_peft_model(tiny_qwen(), LoraConfig(
            task_type='CAUSAL_LM', r=2, lora_alpha=2, target_modules=['q_proj']))
        expected = {name.removeprefix('base_model.model.').replace('.base_layer', '')
                    for name, _ in model.named_parameters() if '.lora_' not in name}
        self.assertEqual(set(dict(iter_policy_weights(model))), expected)

    def test_inverse_mapping_collision_is_rejected(self):
        model = get_peft_model(tiny_gemma(True).to(torch.bfloat16), LoraConfig(
            task_type='CAUSAL_LM', r=2, lora_alpha=2, target_modules=['patch_dense']))
        backbone = model.get_base_model().model
        backbone.vision_embedder = torch.nn.Module()
        backbone.vision_embedder.register_parameter(
            'pos_embedding', torch.nn.Parameter(torch.ones_like(backbone.embed_vision.pos_embedding)))
        with self.assertRaisesRegex(ValueError, 'Duplicate rollout weight name'):
            list(iter_policy_weights(model))


if __name__ == '__main__':
    unittest.main()
