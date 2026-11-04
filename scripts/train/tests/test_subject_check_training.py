import json
import unittest
from scripts.train.shared import config

KEYS = ('subject_is_human_or_human_like_manikin', 'subject_is_not_human_or_manikin')
def answer(positive):
    return json.dumps(dict(zip(KEYS, (int(positive), int(not positive)))))

class SubjectTrainingTests(unittest.TestCase):
    def test_config_module_entrypoint_keeps_optional_validator_available(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('{}')
            result = subprocess.run([sys.executable, '-m', 'scripts.train.shared.config', '--config', str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_opt_in_requires_explicit_modality(self):
        validate = getattr(config, 'validate_subject_check', None)
        self.assertIsNotNone(validate)
        validate({})
        validate({'subject_check': {'enabled': False}})
        with self.assertRaises(ValueError):
            validate({'subject_check': {'enabled': True}})
        for modality in ('rgb', 'ir', 'both'):
            validate({'subject_check': {'enabled': True, 'modality': modality}})
        for bad in ('RGB', None, 'auto'):
            with self.assertRaises(ValueError):
                validate({'subject_check': {'enabled': True, 'modality': bad}})

    def test_metrics_report_invalid_and_positive_recall_by_modality(self):
        from scripts.train.shared import subject_check_evaluation as subject
        rows = [dict(answer=answer(a), prediction=p, reference={'modality': m}) for a,p,m in
                [(True, answer(True), 'rgb'), (False, answer(True), 'rgb'),
                 (False, answer(False), 'ir'), (True, '{bad', 'ir')]]
        report = subject.score_subject_check(rows)
        self.assertEqual(report['metrics']['accuracy'], .5)
        self.assertEqual(report['metrics']['false_positive_rate'], .5)
        self.assertEqual(report['metrics']['positive_recall'], .5)
        self.assertEqual(report['metrics']['negative_recall'], .5)
        self.assertEqual(report['modalities']['ir']['invalid_rate'], .5)
        for invalid in ('{"%s":1,"%s":1}' % KEYS, answer(True)[:-1]+',"extra":0}',
                        '{"%s":true,"%s":false}' % KEYS):
            row = dict(rows[0], prediction=invalid)
            self.assertEqual(subject.score_subject_check([row])['metrics']['accuracy'], 0)

    def test_append_preserves_sampler_groups_and_heldout_medical(self):
        from scripts.train.shared.subject_check_evaluation import attach_subject_check
        splits = {'train': [{'task':'medical_vqa'}], 'validation': [{'medical':1}], 'test': []}
        subjects = {'train': [{'task':'subject_check'}], 'validation': [{'heldout':1}], 'test': []}
        original = {'fixed':[0], 'groups':[]}
        updated, plan, report = attach_subject_check(splits, original, None, subjects, {'hash':'abc'})
        self.assertEqual(plan['fixed'], [0,1])
        self.assertEqual(original['fixed'], [0])
        self.assertEqual(updated['validation'], [{'medical':1}])
        self.assertEqual(report['subject_check_evaluation']['validation'], [{'heldout':1}])
        self.assertEqual(report['subject_check']['hash'], 'abc')

class SubjectLaunchTests(unittest.TestCase):
    def test_run_compilation_preserves_opt_in(self):
        from scripts.train.tests.test_cot_integration import run_config
        from scripts.train.shared.run_config import compile_run
        run = run_config()
        run['subject_check'] = {'enabled':True,'modality':'ir','image_processor_kwargs':{}}
        self.assertEqual(compile_run(run)['experiment']['subject_check'], run['subject_check'])

    def test_subject_derivation_receives_clips_before_medical_category_filter(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from scripts.train.sft.launch import _prepare_sft_data
        raw = {s:[{'clip':'kept'}, {'clip':'excluded_category'}] for s in ('train','validation','test')}
        selected = {s:rows[:1] for s,rows in raw.items()}
        captured = []
        def load(media, split, config, medical, *args):
            captured.append(medical)
            return {s:[] for s in raw}, {}
        with patch('scripts.train.sft.launch.load_splits',return_value=raw), \
             patch('scripts.train.sft.launch.select_medical_examples',return_value=selected), \
             patch('scripts.train.shared.subject_check.load_subject_check_splits',side_effect=load):
            _prepare_sft_data(NS(media_root='m',split_dir='s',catalog='c',prompts='p'),
                              {'subject_check':{'enabled':True,'modality':'rgb'}})
        self.assertEqual(captured, [raw])

    def test_ao_and_cot_entrypoints_append_fixed_rows_and_keep_heldout_separate(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from scripts.train.sft.launch import _prepare_sft_data
        medical = {s:[{'task':'medical_vqa', 'answer':'{}'}] for s in ('train','validation','test')}
        subjects = {s:[{'task':'subject_check','answer':answer(s == 'train')}] for s in medical}
        args = NS(media_root='media',split_dir='splits',catalog='catalog',prompts='prompts')
        for fmt in ('answer_only','cot'):
            with patch('scripts.train.sft.launch.load_splits',return_value=medical), \
                 patch('scripts.train.sft.launch.select_medical_examples',side_effect=lambda s,c:s), \
                 patch('scripts.train.sft.launch.read_yaml',return_value={}), \
                 patch('scripts.train.shared.subject_check.load_subject_check_splits',return_value=(subjects,{'hash':'pinned'})), \
                 patch('scripts.train.shared.cot_data.build_cot_splits',return_value=(medical,{'sampling_plan':{'fixed':[0],'groups':[]}})), \
                 patch('scripts.train.shared.cot_data.pin_split_membership'), \
                 patch('scripts.train.shared.subject_check_evaluation.pin_subject_check_manifest'):
                splits, plan, report = _prepare_sft_data(args, {'answer_format':fmt,'subject_check':{'enabled':True,'modality':'both'}})
            self.assertEqual([r['task'] for r in splits['train']], ['medical_vqa','subject_check'])
            self.assertEqual(plan['fixed'], [0,1])
            self.assertEqual(splits['validation'], medical['validation'])
            self.assertEqual(report['subject_check_evaluation']['test'], subjects['test'])

class SubjectManifestTests(unittest.TestCase):
    def test_subject_addition_changes_reviewed_cot_release_manifest(self):
        from scripts.train.shared import subject_check_evaluation as subject
        from scripts.train.shared.cot_data import fingerprint
        provenance = {'manifest_sha256':'a'*64, 'summary_sha256':'b'*64,
                      'split_sha256':{s:str(i)*64 for i,s in enumerate(('train','validation','test'))},
                      'prompt_sha256':'c'*64}
        report = {'manifest':{'schema_version':1,'files':{'medical':'m'},'sha256':fingerprint({'medical':'m'})},
                  'source_paths':{'files':{}}, 'subject_check':provenance}
        old = dict(report['manifest'])
        subject.pin_subject_check_manifest(report, '/media', '/splits')
        self.assertNotEqual(report['manifest'], old)
        self.assertEqual(report['manifest']['files']['subject_check/train.jsonl'], '0'*64)
        self.assertEqual(report['source_paths']['files']['subject_check/train.jsonl'], '/splits/subject_check/train.jsonl')
        self.assertEqual(report['manifest']['subject_check_prompt_sha256'], 'c'*64)

class SubjectBoundaryTests(unittest.TestCase):
    def test_configured_native_budgets_reach_image_processor(self):
        from scripts.train.shared.config import validate_subject_check
        from scripts.train.shared.collator import AnswerOnlyCollator
        from scripts.train.tests.test_subject_image_collator import ImageProcessorBoundary, ROW
        for family, budget in [('qwen', {'size':{'shortest_edge':1024,'longest_edge':65536}}),
                               ('gemma', {'max_soft_tokens':280})]:
            policy = {'enabled':True,'modality':'rgb','image_processor_kwargs':budget}
            validate_subject_check({'subject_check':policy})
            processor = ImageProcessorBoundary(family)
            AnswerOnlyCollator(processor, 5, 100, image_processor_kwargs=policy['image_processor_kwargs']).prompt_batch([ROW])
            self.assertEqual(processor.calls[0][1]['processor_kwargs']['images_kwargs'], budget)

    def test_mixed_dataset_preserves_image_columns_and_video_discriminator(self):
        from datasets import Dataset
        from scripts.train.shared.subject_check_evaluation import attach_subject_check
        splits = {'train':[{'video':'v.mp4','answer':'{}'}], 'validation':[], 'test':[]}
        subjects = {'train':[{'task':'subject_check','media_type':'image','image':'i.jpg','answer':answer(False)}], 'validation':[], 'test':[]}
        updated, _, _ = attach_subject_check(splits, None, None, subjects, {})
        dataset = Dataset.from_list(updated['train'])
        self.assertEqual(dataset[0]['media_type'], 'video')
        self.assertEqual(dataset[0]['task'], 'medical_vqa')
        self.assertEqual(dataset[1]['image'], 'i.jpg')
        self.assertIsNone(dataset[1]['video'])

    def test_ao_baseline_does_not_request_cot_capability_but_subject_disables_thinking(self):
        from types import SimpleNamespace as NS
        from scripts.train.shared.evaluation import MedicalVLLMEvaluator
        requests = []
        def generate(packets, **kwargs):
            requests.append(kwargs)
            return {'completion_ids':[[1]]}
        generation = dict(temperature=0,top_p=1,top_k=0,repetition_penalty=1,min_p=0)
        evaluator = MedicalVLLMEvaluator(NS(generate=generate), {}, NS(tokenizer=NS(decode=lambda *a,**k:'{}')), {}, generation, 'run', subject_reasoning_parser='qwen3')
        evaluator.predict_encoded({'token_ids':[], 'features':{}}, 'v', 1, {})
        self.assertNotIn('enable_thinking', requests[-1])
        evaluator.predict_encoded({'task':'subject_check','packet':{'token_ids':[], 'features':{}}}, 'v', 1, {})
        self.assertIs(requests[-1]['enable_thinking'], False)
        self.assertEqual(requests[-1]['reasoning_parser'], 'qwen3')

class SubjectEvaluatorTests(unittest.TestCase):
    def test_evaluator_scores_subjects_without_changing_medical_score(self):
        import tempfile
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from scripts.train.shared.evaluation import MedicalVLLMEvaluator
        definition = {'subcategories': {'observables': {'cues':[{'name':'cue'}], 'selection_rule':'multi_select_or_none'}}}
        medical = {'answer':'{"cue":1}', 'reference':{'category':'medical','variant':'native'}}
        subject = dict(task='subject_check', answer=answer(False), reference={'modality':'ir'})
        evaluator = MedicalVLLMEvaluator(None, {}, None, {'categories':{'medical':definition}}, {}, 'run',
                                        subject_check_rows={'test':[subject]})
        evaluator.predict_many = lambda trainer, rows, *args: [answer(True) if r.get('task') == 'subject_check' else '{"cue":1}' for r in rows]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = NS(state=NS(global_step=0), accelerator=NS(process_index=0,num_processes=1),
                         args=NS(seed=1,output_dir=tmp), is_world_process_zero=lambda:False,log=lambda m:None)
            metrics = evaluator._evaluate(trainer,[medical],'test',0,None,[],{})
        self.assertEqual(metrics['test_macro_f1'], 1)
        self.assertEqual(metrics['test_subject_check/false_positive_rate'], 1)
        self.assertNotIn('test_subject_check/positive_recall', metrics)

    def test_cot_subject_budget_and_policy_are_direct(self):
        from scripts.train.shared.cot_budget import generation_policy
        config = {'cot':{'thinking_token_budget':1,'medical_max_new_tokens':2048,'caption_max_new_tokens':50},
                  'evaluation':{'max_new_tokens':64}, 'subject_check':{'enabled':True,'modality':'rgb'}}
        policy = generation_policy(config, reasoning_parser='qwen3')
        self.assertEqual(policy['subject_check'], {'thinking':False,'constraint':'task_json_schema','max_new_tokens':64})

if __name__ == '__main__':
    unittest.main()
