"""CPU inspection exercises the same data and processor routes as SFT."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from scripts.train.tests.test_cot_data import fixture
from scripts.train.tests.test_cot_format import METADATA, MODELS
from scripts.train.tests.test_cot_integration import run_config
from scripts.train.shared.run_config import load_run, compile_run


class CoTInspectionTests(unittest.TestCase):
    def test_cpu_cli_exports_student_targets_coverage_hashes_and_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); _,train=fixture(root); _,held=fixture(root,clip_id='held',targets=False)
            for split,rows in [('train',train),('validation',held),('test',[])]:
                (root/(split+'.jsonl')).write_text('\n'.join(json.dumps(r['reference']) for r in rows))
            path=root/'run.yaml'; path.write_text(yaml.safe_dump(run_config()))
            output=root/'inspection'
            command=[sys.executable,'-m','scripts.train.shared.cot_inspect','--config',str(path),'--media-root',str(root),
                     '--split-dir',str(root),'--processor-metadata',str(METADATA/'qwen3.5-9b'),'--output-dir',str(output)]
            completed=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            report=json.loads((output/'inspection.json').read_text())
            self.assertEqual(report['answer_format'],'cot')
            self.assertTrue(report['coverage']['release_ready'])
            self.assertFalse(report['launch_ready'])
            self.assertEqual({r['task'] for r in report['examples']},{'medical_vqa','scene_description','camera_view'})
            medical=next(r for r in report['examples'] if r['task']=='medical_vqa')
            self.assertIn('</think>',medical['supervised_text_private'])
            self.assertNotIn('PRIVATE TRANSCRIPT',medical['student_input'])
            self.assertNotIn('The view does not support',medical['student_input'])
            self.assertGreater(medical['context']['prompt_tokens'],0)
            self.assertTrue(report['generation_budget_checks']['all_covered_targets_fit'])
            self.assertTrue((output/'source-manifest.json').is_file())
            self.assertTrue((output/'configuration.json').is_file())
            self.assertIn('Private supervised target',(output/'inspection.html').read_text())
            manifest=json.loads((output/'source-manifest.json').read_text())
            self.assertTrue(manifest['files'])

    def test_large_model_configs_preserve_agreed_sixteen_rank_recipe(self):
        from scripts.train.tests.test_cot_data import REPO
        from scripts.train.sft.launch import resolve_training_args
        expected={'qwen3.8-27b':('sdpa','1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
                  'gemma4-31b-it':('flex_attention','842da3794eaa0b77d5f08bae87a17459d91ff475')}
        for model,(attention,revision) in expected.items():
            raw=yaml.safe_load((REPO/'scripts/train/configs/cot'/(model+'.yaml')).read_text())
            raw['paths']['media_root']='/explicit/release';raw['paths']['split_dir']='/explicit/split'
            plan=compile_run(raw)
            self.assertEqual(plan['run']['execution']['training_gpus'],16)
            self.assertTrue(plan['run']['execution']['vllm_gpu_sharing'])
            self.assertEqual(plan['experiment']['attention'],attention)
            self.assertEqual(plan['run']['model_settings']['revision'],revision)
            self.assertEqual(plan['experiment']['effective_batch_size'],16)
            self.assertEqual(resolve_training_args(plan['experiment'],16)['gradient_accumulation_steps'],1)

    def test_four_editable_configs_compile_without_choosing_dataset_release(self):
        from scripts.train.tests.test_cot_data import REPO
        for model in MODELS:
            path=REPO/'scripts/train/configs/cot'/(model+'.yaml')
            self.assertTrue(path.is_file(),f'Missing CoT template for {model}')
            raw=yaml.safe_load(path.read_text())
            self.assertIsNone(raw['paths']['media_root'])
            self.assertIsNone(raw['paths']['split_dir'])
            with self.assertRaisesRegex(ValueError,'explicit strings'):
                load_run(path)
            raw['paths']['media_root']='/explicit/release'
            raw['paths']['split_dir']='/explicit/split'
            self.assertEqual(compile_run(raw)['experiment']['answer_format'],'cot')

if __name__=='__main__': unittest.main()
