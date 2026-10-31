"""Optional absent Gemma audio must not prevent FSDP text/vision wrapping."""
import tempfile
import unittest
from accelerate import FullyShardedDataParallelPlugin
from scripts.train.tests.test_video_frame_compat import tiny_gemma
from scripts.train.shared.modeling import load_lora_model


def policy():
    return FullyShardedDataParallelPlugin(
        fsdp_version=2, auto_wrap_policy='transformer_based_wrap',
        cpu_ram_efficient_loading=False,
    )


class GemmaFSDPPolicyTest(unittest.TestCase):
    def test_real_no_audio_policy_preserves_every_text_and_vision_block(self):
        for unified in (False, True):
            with self.subTest(unified=unified), tempfile.TemporaryDirectory() as directory:
                source = tiny_gemma(unified)
                original = set(source._no_split_modules)
                source.save_pretrained(directory)
                model, _ = load_lora_model(
                    {'local_path': directory}, source.config,
                    {'attention': 'sdpa', 'lora': {'r': 2, 'lora_alpha': 4, 'lora_dropout': 0.0}},
                )
                resolved = policy()
                try:
                    resolved.set_auto_wrap_policy(model)
                except ValueError as error:
                    self.fail(f'Optional absent audio prevents real FSDP policy resolution: {error}')
                classes = resolved.auto_wrap_policy.keywords['transformer_layer_cls']
                expected = ({'Gemma4UnifiedTextDecoderLayer'} if unified else
                            {'Gemma4TextDecoderLayer', 'Gemma4VisionEncoderLayer', 'Gemma4VisionPatchEmbedder'})
                self.assertEqual({cls.__name__ for cls in classes}, expected)
                for module in model.modules():
                    if type(module).__name__ in expected:
                        self.assertTrue(resolved.auto_wrap_policy(module=module, recurse=False, nonwrapped_numel=0))
                self.assertEqual(set(source._no_split_modules), original)
                self.assertEqual(set(tiny_gemma(unified)._no_split_modules), original)
                # Unknown explicit requests must still fail rather than silently
                # losing a required model block from sharding/checkpointing.
                explicit = policy()
                explicit.transformer_cls_names_to_wrap = ['NotARealTransformerBlock']
                with self.assertRaisesRegex(ValueError, 'NotARealTransformerBlock'):
                    explicit.set_auto_wrap_policy(model)


if __name__ == '__main__':
    unittest.main()
