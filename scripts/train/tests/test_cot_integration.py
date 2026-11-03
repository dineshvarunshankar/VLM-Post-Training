"""Opt-in routing, held-out teacher coverage, and final-answer constraints."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.train.shared.run_config import load_run, compile_run
from scripts.train.tests.test_cot_data import fixture, REPO


def run_config():
    run=load_run(REPO/'scripts/train/configs/runs/qwen9b_sft_native_stabilized_medical_camera_viewpoint.yaml')
    run['answer_format']='cot'
    run['shared']['cot']={'medical_max_new_tokens':4096,'thinking_token_budget':3064,'caption_max_new_tokens':1024,'caption_review_samples':4,'accepted_implementation_fingerprints':[], 'release_manifest':None, 'release_manifest_sha256':None}
    return run


class CoTIntegrationTests(unittest.TestCase):
    def test_opt_in_config_and_paths_keep_answer_only_unchanged(self):
        baseline=load_run(REPO/'scripts/train/configs/runs/qwen9b_sft_native_stabilized_medical_camera_viewpoint.yaml')
        before=compile_run(baseline)
        self.assertNotIn('answer_format',before['experiment'])
        try: result=compile_run(run_config())
        except ValueError as error: self.fail(f'Opt-in CoT config was rejected: {error}')
        self.assertEqual(result['experiment']['answer_format'],'cot')
        self.assertIn('/cot/sft_only/',result['stages'][0]['output'])
        self.assertEqual(compile_run(baseline),before)
        for recipe in ('rl_only','sft_then_rl'):
            run=run_config(); run['recipe']=recipe
            with self.assertRaisesRegex(ValueError,'CoT.*SFT'):
                compile_run(run)

    def test_cot_forbids_unrelated_auxiliaries_and_stabilized_camera(self):
        for mutate in (lambda r:r['mixture']['counting'].update(enabled=True),lambda r:r['mixture']['camera_view'].update(sources=['stabilized'])):
            run=run_config(); mutate(run)
            with self.assertRaisesRegex(ValueError,'CoT'):
                compile_run(run)

    def test_real_launcher_loader_attaches_targets_and_reports_missing(self):
        from scripts.train.sft import launch
        self.assertTrue(hasattr(launch,'prepare_sft_data'),'Launcher has no CoT data route')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); _,train=fixture(root); _,held=fixture(root,clip_id='held',targets=False)
            for split,rows in [('train',train),('validation',held),('test',[])]:
                (root/(split+'.jsonl')).write_text('\n'.join(json.dumps(r['reference']) for r in rows))
            args=SimpleNamespace(media_root=root,split_dir=root,catalog=REPO/'scripts/train/configs/catalogs/RGB/label_catalog.yaml',prompts=REPO/'scripts/train/prompts/answer_only/RGB/medical_vqa.yaml')
            splits,plan,report=launch.prepare_sft_data(args,compile_run(run_config())['experiment'])
            self.assertTrue(report['release_ready'],report['issues'])
            self.assertEqual(report['task_counts'],{'medical_vqa':2,'scene_description':1,'camera_view':1})
            self.assertEqual(report['teacher_loss']['validation']['covered'],0)
            self.assertEqual(len(splits['validation']),2)
            self.assertEqual(len(plan['fixed']),4)

    def test_generation_constrains_final_answer_keeps_special_tokens_and_finish_reason(self):
        try: module=importlib.import_module('scripts.train.shared.cot_evaluation')
        except ModuleNotFoundError: module=None
        self.assertIsNotNone(module,'CoT evaluator is not implemented')
        from scripts.train.shared.config import evaluation_settings
        from scripts.train.shared.cot_format import extract_cot_answer
        settings=evaluation_settings(compile_run(run_config())['experiment'],{'generation':{'do_sample':False,'temperature':0.0,'top_p':1.0,'top_k':0,'min_p':0.0,'presence_penalty':0.0,'repetition_penalty':1.0}})
        class Client:
            def generate(self,packets,**kwargs):
                self.request=kwargs
                return {'completion_ids':[[1,2]],'finish_reasons':['stop']}
        class Tokenizer:
            def decode(self,ids,*,skip_special_tokens):
                if skip_special_tokens: raise AssertionError('Boundary tokens discarded')
                return 'Observed movement.\n</think>\n\n{"injury":1}<|im_end|>'
        client=Client()
        evaluator=module.CoTVLLMEvaluator(client,{},SimpleNamespace(tokenizer=Tokenizer()),{'categories':{}},settings['generation'],'run',model_family='qwen3_5',experiment=compile_run(run_config())['experiment'])
        schema={'type':'object','properties':{'injury':{'type':'integer','enum':[0,1]}},'required':['injury'],'additionalProperties':False}
        prediction=evaluator.predict_encoded({},'v',42,schema)
        self.assertEqual(client.request['schemas'],[schema])
        self.assertTrue(client.request['enable_thinking'])
        self.assertEqual(client.request['max_tokens'],4096)
        self.assertEqual(prediction['answer'],'{"injury":1}')
        self.assertEqual(prediction['diagnostics']['finish_reason'],'stop')

    def test_rollout_client_retains_server_finish_reason(self):
        from scripts.train.rl.rollout.client import VideoRolloutClient
        client=VideoRolloutClient('http://unused','model')
        client._checked=True
        packet={'token_ids':[1000],'features':{'mm_placeholders':{}}}
        response={'prompt_token_ids':[1000], 'mm_placeholders':{}, 'choices':[{'index':0,'finish_reason':'length','token_ids':[7], 'logprobs':{'content':[{'token_id':7,'logprob':-1.0}]}}]}
        with patch.object(client,'_request',return_value=response):
            result=client.generate([packet],policy_version='v',temperature=0,top_p=1,top_k=0,repetition_penalty=1,max_tokens=2)
        self.assertEqual(result.get('finish_reasons'),['length'])

    def test_generation_clears_gold_frame_sampling(self):
        import torch
        from scripts.train.shared.cot_evaluation import CoTVLLMEvaluator
        run=compile_run(run_config())
        evaluator=CoTVLLMEvaluator(None,{},None,{},run['run']['model_settings']['generation'],'run',model_family='qwen3_5',experiment=run['experiment'])
        captured=[]
        def prepare(collator,rows):
            captured.extend(rows)
            return {'attention_mask':torch.ones((1,2))}
        trainer=SimpleNamespace(video_collator=SimpleNamespace(max_length=8192))
        row={'task':'medical_vqa','question':'Use the visible video.','reasoning':'PRIVATE','target_identity':{'private':True},'required_frame_indices':[9,20]}
        with patch('scripts.train.shared.cot_evaluation.prepare_video_prompt',side_effect=prepare), patch('scripts.train.shared.cot_evaluation.encode_video_features',return_value={}):
            evaluator.encode(trainer,row,'version')
        self.assertEqual(captured[0]['required_frame_indices'],[])
        self.assertIsNone(captured[0]['reasoning'])
        self.assertEqual(row['required_frame_indices'],[9,20])

    def test_no_teacher_loss_still_scores_all_human_gt(self):
        from scripts.train.sft.trainer import MedicalSFTTrainer
        trainer=object.__new__(MedicalSFTTrainer)
        import torch
        trainer.model=torch.nn.Linear(1,1)
        trainer.eval_dataset=[{'task':'medical_vqa','teacher_available':False,'reasoning':None,'answer':'{"injury":1}'}]
        trainer.video_collator=SimpleNamespace(answer_format='cot')
        captured=[]
        def evaluator(trainer,dataset,prefix,additional_metrics=None):
            captured.append((dataset,additional_metrics))
            return {prefix+'_macro_f1':1.0}
        trainer.medical_evaluator=evaluator
        trainer.get_eval_dataloader=lambda dataset: (_ for _ in ()).throw(AssertionError('Held-out rows without teachers entered loss'))
        result=trainer.evaluate()
        self.assertEqual(result,{'eval_macro_f1':1.0})
        self.assertEqual(len(captured[0][0]),1)
        self.assertEqual(captured[0][1]['eval_teacher_loss_covered'],0)

if __name__=='__main__': unittest.main()
