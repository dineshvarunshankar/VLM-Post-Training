from types import SimpleNamespace
import unittest
from scripts.train.rl.rollout.video_profile import profiling_flags


class SubjectImageProfileTests(unittest.TestCase):
    def test_subject_images_enabled_without_changing_video_budget(self):
        processor = SimpleNamespace(
            video_processor=type('Gemma4VideoProcessor', (), {'patch_size':16, 'pooling_kernel_size':3})(),
            image_processor=type('Gemma4ImageProcessor', (), {})())
        video = {'gemma': {'max_soft_tokens':140}}
        args = dict(profile_video_frames=50, model_config=SimpleNamespace())
        baseline = profiling_flags(processor, video, **args)
        enabled = profiling_flags(processor, video, subject_check={'enabled': True,
                                  'image_processor_kwargs': {'max_soft_tokens':280}}, **args)
        self.assertEqual(baseline['limit_mm_per_prompt']['image'], 0)
        self.assertEqual(enabled['limit_mm_per_prompt']['image'], 1)
        self.assertEqual(enabled['limit_mm_per_prompt']['video'], baseline['limit_mm_per_prompt']['video'])
        self.assertEqual(enabled['mm_processor_kwargs']['videos_kwargs'], baseline['mm_processor_kwargs']['videos_kwargs'])
        self.assertEqual(enabled['mm_processor_kwargs']['images_kwargs'], {'max_soft_tokens':280})

if __name__ == '__main__': unittest.main()
