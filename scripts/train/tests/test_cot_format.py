"""Native templates, strict final boundaries, and real processor masking."""
import importlib
import os
from pathlib import Path
import tempfile
import unittest

from scripts.train.tests.test_cot_data import fixture

METADATA = Path(os.environ.get('COT_TEMPLATE_ROOT','/home/ubuntu/triage/scratch/vlm-benchmark/memory-estimation/metadata'))
MODELS = ('qwen3.5-9b','qwen3.8-27b','gemma4-12b-it','gemma4-31b-it')


def cot_module(test):
    try: module=importlib.import_module('scripts.train.shared.cot_format')
    except ModuleNotFoundError: module=None
    test.assertIsNotNone(module, 'Native CoT formatting is not implemented')
    return module


class CoTParsingTests(unittest.TestCase):
    def test_final_json_comes_after_native_boundary(self):
        module=cot_module(self)
        samples = {
            'qwen3_5':'The object {"injury":0} is a reasoning example.\n</think>\n\n{"injury":1}<|im_end|>\n',
            'gemma4':'<|channel>thought\nThe object {"injury":0} is a reasoning example.\n<channel|>{"injury":1}<turn|>\n',
        }
        for family,text in samples.items():
            result=module.extract_cot_answer(text,model_family=family)
            self.assertEqual(result.answer,'{"injury":1}')
            self.assertFalse(result.diagnostics['malformed'])
            self.assertFalse(result.diagnostics['empty_reasoning'])

    def test_control_boundaries_allow_generated_whitespace_variations(self):
        extract=cot_module(self).extract_cot_answer
        for family,text in [('qwen3_5','Visible cue.</think>{"injury":1}<|im_end|>'),('gemma4','<|channel>thought\nVisible cue.<channel|>{"injury":1}<turn|>')]:
            self.assertEqual(extract(text,model_family=family).answer,'{"injury":1}')

    def test_missing_boundary_multiple_answers_and_truncation_are_invalid(self):
        extract=cot_module(self).extract_cot_answer
        for text in ('{"injury":1}<|im_end|>', 'reason\n</think>\n\n{"injury":1}{"injury":1}<|im_end|>', 'reason\n</think>\n\n{"injury":1}', 'reason\n</think>\n\n{"injury":1,"injury":0}<|im_end|>'):
            result=extract(text,model_family='qwen3_5')
            self.assertIsNone(result.answer,text)
            self.assertTrue(result.diagnostics['malformed'])
        result=extract('reason\n</think>\n\n{"injury":1}',model_family='qwen3_5',finish_reason='stop')
        self.assertEqual(result.answer,'{"injury":1}')
        result=extract('reason\n</think>\n\n{"injury":1}<|im_end|>',model_family='qwen3_5',finish_reason='length')
        self.assertIsNone(result.answer)
        self.assertTrue(result.diagnostics['truncated'])


class CoTNativeMaskTests(unittest.TestCase):
    def test_medical_and_direct_targets_on_all_pinned_processors(self):
        module=cot_module(self)
        from transformers import AutoConfig, AutoProcessor
        from scripts.train.shared.collator import CoTCollator
        from scripts.train.shared.video import spatial_options
        with tempfile.TemporaryDirectory() as directory:
            _,rows=fixture(Path(directory),targets=False)
            medical=dict(rows[0],task='medical_vqa',reasoning='The view is inadequate. An example JSON {"wrong":0} is not the final answer.',teacher_available=True)
            caption=dict(rows[0],task='scene_description',reasoning=None,answer='A person is visible.',teacher_available=True)
            camera=dict(rows[0],task='camera_view',reasoning=None,answer='{"aerial_top_down":1,"aerial_oblique":0}',teacher_available=True)
            for name in MODELS:
                with self.subTest(model=name):
                    self.assertTrue((METADATA/name/'chat_template.jinja').is_file(),f'Missing pinned template: {name}')
                    processor=AutoProcessor.from_pretrained(METADATA/name,local_files_only=True)
                    family=module.model_family(AutoConfig.from_pretrained(METADATA/name,local_files_only=True))
                    video={'fps':2,'qwen':{'max_pixels':65536},'gemma':{'max_soft_tokens':70}}
                    collator=CoTCollator(processor,2,8192,spatial_options(processor,video),model_family=family,video_config=video)
                    collator.medical_token_metrics = True
                    for example in (medical,caption,camera):
                        batch=collator([example])
                        ids=batch['input_ids'][0]; labels=batch['labels'][0]
                        text=processor.tokenizer.decode(ids[labels!=-100],skip_special_tokens=False)
                        prompt=processor.tokenizer.decode(ids[(labels==-100)&batch['attention_mask'][0].bool()],skip_special_tokens=False)
                        self.assertIn(example['question'],prompt)
                        self.assertNotIn(example['question'],text)
                        self.assertIn(example['answer'],text)
                        self.assertTrue(text.rstrip().endswith('<|im_end|>' if family=='qwen3_5' else '<turn|>'))
                        if example['task']=='medical_vqa':
                            groups = batch['medical_token_groups'][0]
                            value_text = processor.tokenizer.decode(ids[groups == 1])
                            self.assertTrue((groups[labels == -100] == 0).all())
                            self.assertNotIn('wrong', value_text)
                            self.assertGreater(int((groups == 1).sum()), 0)
                            self.assertIn(example['reasoning'],text)
                            self.assertEqual(module.extract_cot_answer(text,model_family=family).answer,example['answer'])
                        else:
                            self.assertEqual(int(batch['medical_token_groups'].sum()), 0)
                            self.assertNotIn('The view is inadequate',text)
                            self.assertNotIn('<|channel>thought',text)
                            self.assertNotIn('<think>',text)
                        self.assertTrue((labels[batch['attention_mask'][0]==0]==-100).all())
                    collator.max_length=2
                    with self.assertRaisesRegex(ValueError,'never silently truncated'):
                        collator([medical])

if __name__=='__main__': unittest.main()
