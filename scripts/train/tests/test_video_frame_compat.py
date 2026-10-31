"""Real Gemma video forwards must preserve unequal per-video frame counts."""

import unittest
import tempfile
from functools import wraps

import torch
import transformers
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper,
)
from trl.trainer.utils import add_fused_lm_head
from scripts.train.shared.video_compat import install_video_frame_compatibility
from scripts.train.shared.modeling import load_lora_model


def tiny_gemma(unified):
    prefix = 'Gemma4Unified' if unified else 'Gemma4'
    text = dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=16, global_head_dim=16, max_position_embeddings=64,
                layer_types=['sliding_attention', 'full_attention'], sliding_window=16,
                use_cache=False)
    if not unified:
        # The pinned 31B config disables per-layer input projections.
        text.update(vocab_size_per_layer_input=128, hidden_size_per_layer_input=0)
    vision = (dict(patch_size=2, pooling_kernel_size=3, mm_embed_dim=32,
                   mm_posemb_size=16, output_proj_dims=32) if unified else
              dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=2, head_dim=16,
                   patch_size=2, pooling_kernel_size=3, position_embedding_size=16))
    config = getattr(transformers, prefix + 'Config')(
        text_config=text, vision_config=vision, audio_config=None,
        image_token_id=126, video_token_id=127, audio_token_id=125,
    )
    model = getattr(transformers, prefix + 'ForConditionalGeneration')(config)
    model.config.use_cache = False
    return model


def two_videos(unified):
    ids = torch.tensor([[1, 127, 127, 3, 4, 0, 0], [1, 127, 127, 127, 127, 3, 4]])
    pixels = torch.randn(3, 2, 108) if unified else torch.randn(3, 18, 12)
    positions = [[0, 0], [1, 0]] if unified else [[x, y] for y in range(3) for x in range(6)]
    return dict(input_ids=ids, attention_mask=ids.ne(0).long(),
                mm_token_type_ids=ids.eq(127).long() * 2,
                labels=ids.masked_fill((ids == 127) | (ids == 0) | (ids == 1), -100),
                pixel_values_videos=pixels,
                video_position_ids=torch.tensor([positions] * 3),
                num_frames_per_video=torch.tensor([1, 2]))


class VideoFrameCompatibilityTests(unittest.TestCase):
    def test_production_loader_lora_and_fused_video_backward(self):
        for unified in (True, False):
            with self.subTest(unified=unified), tempfile.TemporaryDirectory() as directory:
                source = tiny_gemma(unified)
                source.save_pretrained(directory)
                model, _ = load_lora_model(
                    {'local_path': directory}, source.config,
                    {'attention': 'sdpa', 'lora': {'r': 2, 'lora_alpha': 4, 'lora_dropout': 0.0}},
                )
                add_fused_lm_head(model.get_base_model())
                output = model(**two_videos(unified), fused_lm_head=True)
                self.assertTrue(torch.isfinite(output.loss))
                output.loss.backward()
                gradients = [parameter.grad for name, parameter in model.named_parameters()
                             if 'lora_' in name and parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients))

    def test_regular_and_fused_video_loss_and_backward(self):
        for unified in (True, False):
            with self.subTest(unified=unified):
                torch.manual_seed(42)
                model = tiny_gemma(unified)
                install_video_frame_compatibility(model)
                batch = two_videos(unified)
                regular = model(**batch).loss
                add_fused_lm_head(model)
                fused = model(**batch, fused_lm_head=True).loss
                torch.testing.assert_close(fused, regular)
                fused.backward()
                vision = model.model.embed_vision if unified else model.model.vision_tower
                gradients = [p.grad for p in vision.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_video_backward_with_checkpoint_recomputation(self):
        for unified in (True, False):
            for checkpoint in ('layers', 'backbone'):
                for fused in (False, True):
                    with self.subTest(unified=unified, checkpoint=checkpoint, fused=fused):
                        torch.manual_seed(42)
                        model = tiny_gemma(unified)
                        install_video_frame_compatibility(model)
                        if checkpoint == 'layers':
                            apply_activation_checkpointing(
                                model, checkpoint_wrapper_fn=checkpoint_wrapper,
                                check_fn=lambda module: type(module).__name__ in model._no_split_modules,
                            )
                        else:
                            model.model = checkpoint_wrapper(model.model)
                        if fused:
                            add_fused_lm_head(model)
                        batch = two_videos(unified)
                        output = model(**batch, fused_lm_head=True) if fused else model(**batch)
                        self.assertTrue(torch.isfinite(output.loss))
                        output.loss.backward()
                        vision = model.model.embed_vision if unified else model.model.vision_tower
                        gradients = [p.grad for p in vision.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_exact_counts_nested_calls_and_exception_cleanup(self):
        for unified in (True, False):
            with self.subTest(unified=unified):
                model = tiny_gemma(unified)
                outer, inner = two_videos(unified), two_videos(unified)
                original_features = model.model.get_video_features
                original_text_forward = model.model.language_model.forward

                @wraps(original_text_forward)
                def text_forward(*args, **kwargs):
                    self.assertNotIn('num_frames_per_video', kwargs)
                    self.assertNotIn('_triage_video_frames', kwargs)
                    return original_text_forward(*args, **kwargs)

                model.model.language_model.forward = text_forward
                seen = []
                nested = [True]
                fail = [False]

                @wraps(original_features)
                def observed(*args, **kwargs):
                    seen.append(kwargs.get('num_frames_per_video'))
                    if fail[0]:
                        raise RuntimeError('injected feature failure')
                    if nested[0]:
                        nested[0] = False
                        model(**inner)
                        model.model.get_video_features(
                            outer['pixel_values_videos'], outer['video_position_ids'],
                        )
                    result = original_features(*args, **kwargs)
                    self.assertEqual([part.shape[0] for part in result.pooler_output], [2, 4])
                    return result

                model.model.get_video_features = observed
                install_video_frame_compatibility(model)
                model(**outer)
                self.assertEqual(len(seen), 3)
                for actual, expected in zip(seen, (outer, inner, outer)):
                    self.assertIs(actual, expected['num_frames_per_video'])
                add_fused_lm_head(model)
                model(**outer, fused_lm_head=True)
                self.assertIs(seen[-1], outer['num_frames_per_video'])
                fail[0] = True
                with self.assertRaisesRegex(RuntimeError, 'injected feature failure'):
                    model(**inner, fused_lm_head=True)
                fail[0] = False
                missing = {key: value for key, value in outer.items() if key != 'num_frames_per_video'}
                with self.assertRaisesRegex(ValueError, 'num_frames_per_video'):
                    model(**missing, fused_lm_head=True)
                self.assertIsNone(seen[-1])
                with self.assertRaisesRegex(ValueError, 'num_frames_per_video'):
                    model.model.get_video_features(outer['pixel_values_videos'], outer['video_position_ids'])
                self.assertIsNone(seen[-1])
                self.assertTrue(torch.isfinite(model(input_ids=torch.tensor([[1, 3, 4]])).logits).all())


if __name__ == '__main__':
    unittest.main()
