"""CPU image decoding and target masking with the four pinned processors."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from scripts.train.tests.test_subject_image_features import boundary_modules

METADATA = Path(os.environ.get('COT_TEMPLATE_ROOT', '/home/ubuntu/triage/scratch/vlm-benchmark/memory-estimation/metadata'))
MODELS = ('qwen3.5-9b', 'qwen3.8-27b', 'gemma4-12b-it', 'gemma4-31b-it')


@unittest.skipUnless(all((METADATA / name / 'chat_template.jinja').is_file() for name in MODELS),
                     'Pinned local processor metadata is unavailable')
class NativeImageTests(unittest.TestCase):
    def test_compiled_qwen_budget_reaches_native_image_processor(self):
        from scripts.train.shared.run_config import load_run, compile_run
        from scripts.train.shared.modeling import build_processor
        from scripts.train.rl.video import prepare_video_prompt
        run = load_run(Path(__file__).resolve().parents[1] / 'configs/runs/qwen9b_sft_stabilized_medical.yaml')
        run['subject_check'] = {'enabled': True, 'modality': 'rgb',
            'image_processor_kwargs': {'size': {'shortest_edge': 1024, 'longest_edge': 4096}}}
        plan = compile_run(run)
        profile = dict(plan['run']['model_settings'], local_path=str(METADATA / 'qwen3.5-9b'))
        _, _, collator = build_processor(profile, plan['experiment'])
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'image.jpg'
            Image.new('RGB', (512, 512), color='green').save(image_path)
            row = {'task': 'subject_check', 'media_type': 'image', 'image': str(image_path),
                   'system': 'Classify the target.', 'question': 'Is this target human?'}
            packet = prepare_video_prompt(collator, [row])
            self.assertEqual(packet['image_grid_thw'].tolist(), [[1, 4, 4]])
            self.assertNotIn('pixel_values_videos', packet)

    def test_real_image_processors_mask_direct_targets_in_ao_and_cot(self):
        from transformers import AutoConfig, AutoProcessor
        from scripts.train.shared.collator import AnswerOnlyCollator, CoTCollator
        from scripts.train.shared.cot_format import model_family
        from scripts.train.shared.vllm import feature_configuration
        from scripts.train.rl.video import prepare_video_prompt
        from scripts.train.rl.rollout.features import encode_video_features
        answer = '{"subject_is_human_or_human_like_manikin":0,"subject_is_not_human_or_manikin":1}'
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'image.jpg'
            Image.new('RGB', (128, 96), color=(80, 120, 70)).save(image_path)
            row = {'task': 'subject_check', 'media_type': 'image', 'image': str(image_path),
                   'system': 'Classify the target.', 'question': 'Is this target human?',
                   'answer': answer, 'reasoning': '', 'teacher_available': True}
            for name in MODELS:
                config = AutoConfig.from_pretrained(METADATA / name, local_files_only=True)
                processor = AutoProcessor.from_pretrained(METADATA / name, local_files_only=True)
                family = model_family(config)
                budget = {'size': {'shortest_edge': 1024, 'longest_edge': 65536}} if family == 'qwen3_5' else {'max_soft_tokens': 70}
                for cls in (AnswerOnlyCollator, CoTCollator):
                    with self.subTest(model=name, mode=cls.__name__):
                        options = {'model_family': family} if cls is CoTCollator else {}
                        collator = cls(processor, 5, 8192, image_processor_kwargs=budget, **options)
                        collator.medical_token_metrics = True
                        batch = collator([row])
                        target_ids = batch['input_ids'][batch['labels'] != -100]
                        target = processor.tokenizer.decode(target_ids, skip_special_tokens=False)
                        self.assertEqual(target.strip(), answer + ('<|im_end|>' if family == 'qwen3_5' else '<turn|>'))
                        self.assertEqual(int(batch['medical_token_groups'].sum()), 0)
                        self.assertIn('pixel_values', batch)
                        self.assertNotIn('pixel_values_videos', batch)
                        if cls is CoTCollator:
                            self.assertEqual(batch['cot_task_ids'].tolist(), [3])
                        packet = prepare_video_prompt(collator, [row])
                        with patch.dict(sys.modules, boundary_modules()), patch('scripts.train.rl.rollout.features.version', return_value='0.30.0'):
                            result = encode_video_features(packet, policy_version='cpu:1', **feature_configuration(config, processor))
                        placeholder = result['features']['mm_placeholders']['image'][0]
                        selected = result['token_ids'][placeholder['offset']:placeholder['offset'] + placeholder['length']]
                        self.assertEqual(set(selected), {processor.image_token_id})
                        values = json.loads(result['features']['kwargs_data']['image'][0])['values']
                        self.assertEqual(set(values), {'pixel_values', 'image_grid_thw'} if family == 'qwen3_5' else {'pixel_values', 'pixel_position_ids'})

if __name__ == '__main__': unittest.main()
