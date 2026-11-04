"""Image prompts must retain native image fields without video metadata."""
from types import SimpleNamespace
import unittest
import torch
from scripts.train.shared.collator import AnswerOnlyCollator, CoTCollator
from scripts.train.rl.video import prepare_video_prompt


class ImageProcessorBoundary:
    def __init__(self, family='qwen'):
        name = 'Qwen2VLImageProcessor' if family == 'qwen' else 'Gemma4ImageProcessor'
        self.image_processor = type(name, (), {})()
        self.video_processor = SimpleNamespace()
        self.tokenizer = SimpleNamespace(padding_side='right', pad_token_id=0)
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones(1, 2, dtype=torch.long),
                'pixel_values': torch.ones(4, 8), 'image_grid_thw': torch.tensor([[1, 2, 2]])}


ROW = {'task': 'subject_check', 'media_type': 'image', 'image': '/crop.jpg',
       'system': 's', 'question': 'q', 'answer': '{}', 'reference': {'modality': 'ir'}}


class SubjectImageCollatorTests(unittest.TestCase):
    def test_native_image_prompt_ignores_video_budgets_and_restores_padding(self):
        processor = ImageProcessorBoundary()
        collator = AnswerOnlyCollator(processor, 5, 100,
            processor_kwargs={'videos_kwargs': {'size': {'longest_edge': 999}}},
            video_config={'sources': {'native': {}}})
        packet = prepare_video_prompt(collator, [ROW])
        self.assertIn('pixel_values', packet)
        self.assertNotIn('pixel_values_videos', packet)
        messages, kwargs = processor.calls[0]
        self.assertEqual(messages[0][1]['content'][0], {'type': 'image', 'path': '/crop.jpg'})
        self.assertEqual(kwargs['processor_kwargs'], {'padding': True, 'truncation': False})
        self.assertFalse(kwargs['enable_thinking'])
        self.assertEqual(processor.tokenizer.padding_side, 'right')
        self.assertFalse(hasattr(processor.video_processor, 'sample_frames'))

    def test_image_budget_is_separate_and_family_checked(self):
        for family, budget in [('qwen', {'size': {'shortest_edge': 1024, 'longest_edge': 32768}}),
                               ('gemma', {'max_soft_tokens': 280})]:
            processor = ImageProcessorBoundary(family)
            collator = AnswerOnlyCollator(processor, 5, 100, image_processor_kwargs=budget)
            collator.prompt_batch([ROW])
            self.assertEqual(processor.calls[0][1]['processor_kwargs']['images_kwargs'], budget)
        for family, budget in [('qwen', {'max_soft_tokens': 280}), ('gemma', {'max_soft_tokens': 141}),
                               ('qwen', {'size': {'shortest_edge': 4096, 'longest_edge': 100}}),
                               ('qwen', {'fps': 5}), ('gemma', {'videos_kwargs': {}})]:
            with self.subTest(family=family, budget=budget), self.assertRaises(ValueError):
                AnswerOnlyCollator(ImageProcessorBoundary(family), 5, 100,
                    image_processor_kwargs=budget).prompt_batch([ROW])

    def test_cot_subject_image_disables_thinking(self):
        processor = ImageProcessorBoundary()
        collator = CoTCollator(processor, 5, 100, model_family='qwen3_5')
        collator.prompt_batch([ROW])
        self.assertFalse(processor.calls[0][1]['enable_thinking'])

    def test_nullable_legacy_video_discriminator_remains_video(self):
        row = {"media_type": None, "video": "/video.mp4", "system": "s", "question": "q"}
        self.assertEqual(AnswerOnlyCollator.messages(row)[1]["content"][0],
                         {"type": "video", "path": "/video.mp4"})

    def test_unknown_media_and_mixed_batches_fail_clearly(self):
        collator = AnswerOnlyCollator(ImageProcessorBoundary(), 5, 100)
        for rows in [[dict(ROW, media_type='audio')], [ROW, {'video': '/video.mp4'}]]:
            with self.assertRaises(ValueError):
                collator.prompt_batch(rows)


if __name__ == '__main__': unittest.main()
