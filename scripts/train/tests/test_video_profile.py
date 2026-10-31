"""Profiling flags must preserve the real processor budget and cover full frames."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from transformers import Qwen3VLVideoProcessor, Gemma4VideoProcessor, Gemma4UnifiedVideoProcessor
from scripts.train.rl.rollout.video_profile import profiling_flags
from scripts.train.shared.video import spatial_options


VIDEO = {'fps': 10, 'qwen': {'max_pixels': 251658240}, 'gemma': {'max_soft_tokens': 280}}


class VideoProfileTests(unittest.TestCase):
    def test_qwen_preserves_full_processor_options_without_shrinking_dummy(self):
        processor = SimpleNamespace(video_processor=Qwen3VLVideoProcessor(size={'shortest_edge': 4096, 'longest_edge': 16777216}))
        original = deepcopy(VIDEO)
        result = profiling_flags(processor, VIDEO, profile_video_frames=150, model_config=SimpleNamespace())
        self.assertEqual(result['mm_processor_kwargs'], spatial_options(processor, VIDEO))
        self.assertEqual(result['mm_processor_kwargs']['videos_kwargs'], {
            'size': {'shortest_edge': 4096, 'longest_edge': 251658240}, 'cap_pixels_per_frame': False})
        self.assertEqual(result['limit_mm_per_prompt'], {'image': 0, 'video': {'count': 1}})
        self.assertEqual(VIDEO, original)

    def test_both_gemma_profiles_cover_exact_full_spatial_budget(self):
        for cls in (Gemma4VideoProcessor, Gemma4UnifiedVideoProcessor):
            with self.subTest(processor=cls.__name__):
                processor = SimpleNamespace(video_processor=cls(patch_size=16, pooling_kernel_size=3))
                result = profiling_flags(processor, VIDEO, profile_video_frames=150, model_config=SimpleNamespace())
                self.assertEqual(result['mm_processor_kwargs'], {'videos_kwargs': {'max_soft_tokens': 280}})
                self.assertEqual(result['limit_mm_per_prompt']['video'], {'count': 1, 'num_frames': 150, 'width': 960, 'height': 672})
                self.assertEqual(result['limit_mm_per_prompt']['video']['width'] * result['limit_mm_per_prompt']['video']['height'] // (48 * 48), 280)

    def test_only_checkpoints_with_audio_disable_the_supported_audio_modality(self):
        processor = SimpleNamespace(video_processor=Gemma4UnifiedVideoProcessor())
        for audio, expected in ((None, False), (SimpleNamespace(), True)):
            with self.subTest(audio_supported=expected):
                result = profiling_flags(processor, VIDEO, profile_video_frames=150,
                                         model_config=SimpleNamespace(audio_config=audio))
                self.assertEqual('audio' in result['limit_mm_per_prompt'], expected)
                if expected:
                    self.assertEqual(result['limit_mm_per_prompt']['audio'], 0)

    def test_invalid_frame_bounds_are_rejected(self):
        processor = SimpleNamespace(video_processor=Gemma4VideoProcessor())
        for value in (True, 0, 1, -1, 150.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profiling_flags(processor, VIDEO, profile_video_frames=value, model_config=SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
