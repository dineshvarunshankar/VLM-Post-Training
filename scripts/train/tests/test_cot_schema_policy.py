"""CoT evaluation uses task schemas without constraining medical reasoning."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.train.shared.cot_evaluation import CoTVLLMEvaluator
from scripts.train.shared.cot_budget import generation_policy
from scripts.train.shared.run_config import compile_run
from scripts.train.tests.test_cot_integration import run_config
from scripts.train.rl.rollout.client import VideoRolloutClient

SCHEMA = {'type':'object','properties':{'cue':{'type':'integer','enum':[0,1]}},'required':['cue'],'additionalProperties':False}
CAMERA_SCHEMA = {'type':'object','properties':{
    'aerial_top_down':{'type':'integer','enum':[0,1]},
    'aerial_oblique':{'type':'integer','enum':[0,1]},
    'ground_level':{'type':'integer','enum':[0,1]}},
    'required':['aerial_top_down','aerial_oblique','ground_level'],'additionalProperties':False}
SETTINGS = dict(do_sample=False,temperature=0,top_p=1,top_k=0,min_p=0,repetition_penalty=1,presence_penalty=0)

class CoTSchemaPolicyTests(unittest.TestCase):
    def test_task_requests_for_both_model_families(self):
        for family,parser,raw,ending in (
            ('qwen3_5','qwen3','Visible evidence.</think>{"cue":1}<|im_end|>','<|im_end|>'),
            ('gemma4','gemma4','<|channel>thought\nVisible evidence.<channel|>{"cue":1}<turn|>','<turn|>'),
        ):
            calls=[]
            class Client:
                def generate(self,packets,**kwargs):
                    calls.append(kwargs)
                    return {'completion_ids':[[1]],'finish_reasons':['stop']}
            text=[raw]
            processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a,**k:text[0]))
            evaluator=CoTVLLMEvaluator(Client(),{},processor,{},SETTINGS,'run',model_family=family,experiment=compile_run(run_config())['experiment'])
            for task in ('medical_vqa','camera_view','scene_description'):
                with self.subTest(family=family,task=task):
                    text[0]=raw if task=='medical_vqa' else ('A person is visible.' if task=='scene_description' else '{"cue":1}')+ending
                    schema=None if task=='scene_description' else SCHEMA
                    result=evaluator.predict_encoded({'packet':{},'task':task},'v',42,schema)
                    self.assertFalse(result['diagnostics']['malformed'])
                    request=calls[-1]
                    self.assertEqual(request['schemas'],None if schema is None else [SCHEMA])
                    self.assertEqual(request['enable_thinking'],task=='medical_vqa')
                    self.assertEqual(request['reasoning_parser'],parser)
                    self.assertEqual('thinking_token_budget' in request,task=='medical_vqa')
            for task in ('medical_vqa','camera_view'):
                with self.assertRaisesRegex(ValueError,'schema'):
                    evaluator.predict_encoded({'packet':{},'task':task},'v',42,None)

    def test_row_schemas_use_catalog_and_camera_contract_not_answer(self):
        evaluator=object.__new__(CoTVLLMEvaluator)
        evaluator.catalog={'categories':{'medical':{'subcategories':{'observables':{'cues':[{'name':'cue'}]}}}}}
        medical={'task':'medical_vqa','reference':{'category':'medical'},'answer':'PRIVATE'}
        camera={'task':'camera_view','answer':'PRIVATE'}
        self.assertEqual(evaluator.answer_schema(medical),SCHEMA)
        self.assertEqual(evaluator.answer_schema(camera),CAMERA_SCHEMA)
        self.assertIsNone(evaluator.answer_schema({'task':'scene_description'}))
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            evaluator.answer_schema({'task':'counting'})

    def test_camera_loader_rows_work_without_rl_reward_metadata(self):
        from scripts.train.shared.cot_data import camera_rows
        from scripts.train.shared.data import read_yaml
        from scripts.train.tests.test_cot_data import fixture, REPO
        config=compile_run(run_config())['experiment']
        catalog=read_yaml(REPO/'scripts/train/configs/catalogs/RGB/label_catalog.yaml')
        with tempfile.TemporaryDirectory() as directory:
            _,medical=fixture(Path(directory))
            camera,_=camera_rows(medical,catalog,config)
        self.assertEqual(len(camera),1)
        self.assertNotIn('reward_context',camera[0])
        evaluator=object.__new__(CoTVLLMEvaluator)
        self.assertEqual(evaluator.answer_schema(camera[0]),CAMERA_SCHEMA)

    def test_saved_generation_policy_records_constraints(self):
        config=compile_run(run_config())['experiment']
        for parser in ('qwen3','gemma4'):
            policy=generation_policy(config,reasoning_parser=parser)
            self.assertEqual(policy['medical_vqa']['constraint'],'final_answer_json_schema')
            self.assertEqual(policy['camera_view']['constraint'],'task_json_schema')
            self.assertEqual(policy['scene_description']['constraint'],'none')

    def test_evaluation_routes_schemas_and_reports_medical_selection_separately(self):
        config=compile_run(run_config())['experiment']
        catalog={'categories':{'medical':{'subcategories':{'observables':{'cues':[{'name':'cue'}],'selection_rule':'multi_select_or_none'}}}}}
        evaluator=CoTVLLMEvaluator(None,{},None,catalog,SETTINGS,'run',model_family='qwen3_5',experiment=config)
        medical=[{'reference':{'variant':source,'category':'medical'},'answer':'{"cue":1}'} for source in ('native','stabilized')]
        auxiliary=[{'task':'scene_description','reference':{},'answer':'A person.','teacher_available':True},
                   {'task':'camera_view','reference':{},'answer':'{"aerial_top_down":1,"aerial_oblique":0,"ground_level":0}','teacher_available':True}]
        requests=[]
        def predict(trainer,rows,version,seeds,schemas):
            requests.append(schemas)
            return [{'answer':r['answer'],'raw':r['answer'],'diagnostics':{'output_tokens':3}} for r in rows]
        with tempfile.TemporaryDirectory() as directory:
            trainer=SimpleNamespace(accelerator=SimpleNamespace(process_index=0,num_processes=1),state=SimpleNamespace(global_step=2),
                args=SimpleNamespace(seed=42,output_dir=directory),is_world_process_zero=lambda:True,log=lambda metrics:None)
            with patch.object(evaluator,'predict_many',side_effect=predict), patch('scripts.train.shared.cot_evaluation.gather_object',side_effect=lambda value:value):
                metrics=evaluator._evaluate(trainer,medical,'test',time.monotonic(),None,auxiliary,{})
            report=json.loads((Path(directory)/'test-step-2.json').read_text())
        self.assertEqual(requests,[[SCHEMA,SCHEMA],[None,CAMERA_SCHEMA]])
        self.assertEqual(report['decoding']['constraint'],'final_answer_json_schema')
        self.assertEqual(metrics['test_examples'],2)
        self.assertEqual(metrics['test_invalid_rate'],0)
        self.assertEqual(report['auxiliary']['camera_view']['examples'],1)
        self.assertEqual(report['auxiliary']['scene_description']['examples'],1)

    def test_resume_rejects_old_unconstrained_cot_policy(self):
        from scripts.train.shared.runtime import save_run_metadata
        config=compile_run(run_config())['experiment']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('catalog.yaml','prompts.yaml','train.jsonl'):(root/name).write_text('')
            args=SimpleNamespace(model='qwen3.5-9b',run_location={'answer_format':'cot'},resume=None,
                catalog=root/'catalog.yaml',prompts=root/'prompts.yaml',split_dir=root)
            output=root/'output'
            with patch.dict('os.environ',{'TRIAGE_RUN_CONFIG':''}):
                save_run_metadata(args,output,{'generation':SETTINGS},config,{}, {}, {'train':[]})
                previous=json.loads((output/'run.json').read_text())
                for task in ('medical_vqa','camera_view','scene_description'):
                    previous['generation_budget_policy'][task].pop('constraint')
                (output/'run.json').write_text(json.dumps(previous))
                args.resume=output/'checkpoints/checkpoint-2'
                with self.assertRaisesRegex(ValueError,'Resume configuration'):
                    save_run_metadata(args,output,{'generation':SETTINGS},config,{}, {}, {'train':[]})

    def test_client_forwards_task_thinking_without_changing_default_requests(self):
        for parser in ('qwen3','gemma4'):
            client,requests=self.client(parser)
            for thinking,schema in ((True,SCHEMA),(False,SCHEMA),(False,None)):
                self.generate(client,enable_thinking=thinking,reasoning_parser=parser,schemas=[schema] if schema else None,
                              **({'thinking_token_budget':3064} if thinking else {}))
                body=requests[-1]
                self.assertIs(body['enable_thinking'],thinking)
                self.assertEqual(body['sampling_params'].get('structured_outputs'),{'json':schema} if schema else None)
            self.generate(client)
            self.assertNotIn('enable_thinking',requests[-1])

    def test_client_rejects_unsafe_or_unpatched_server_before_generation(self):
        for change in ('missing_field','reasoning_constrained','wrong_parser'):
            client,requests=self.client('qwen3',change)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'CoT.*schema|CoT.*thinking'):
                self.generate(client,enable_thinking=True,reasoning_parser='qwen3',schemas=[SCHEMA],thinking_token_budget=3064)
            self.assertEqual(requests,[])

    @staticmethod
    def generate(client,**kwargs):
        return client.generate([{'token_ids':[1000],'features':{'mm_placeholders':{}}}],policy_version='v',temperature=0,top_p=1,top_k=0,repetition_penalty=1,max_tokens=4096,**kwargs)

    @staticmethod
    def client(parser,change=None):
        client=VideoRolloutClient('http://unused','model');client._checked=True
        requests=[]
        def request(method,path,**kwargs):
            if path=='/server_info':
                return {'vllm_config':{'reasoning_config':{'reasoning_parser':parser,'_enabled':True,'_reasoning_start_token_ids':[1],'_reasoning_end_token_ids':[2]},'structured_outputs_config':{'reasoning_parser':'wrong' if change=='wrong_parser' else parser,'enable_in_reasoning':change=='reasoning_constrained'}}}
            if path=='/openapi.json':
                return {'paths':{'/inference/v1/generate':{'post':{'requestBody':{'content':{'application/json':{'schema':{'$ref':'#/components/schemas/GenerateRequest'}}}}}}},'components':{'schemas':{'GenerateRequest':{'properties':{} if change=='missing_field' else {'enable_thinking':{'anyOf':[{'type':'boolean'},{'type':'null'}]}}}}}}
            requests.append(kwargs['json'])
            return {'prompt_token_ids':[1000],'mm_placeholders':{},'choices':[{'index':0,'finish_reason':'stop','token_ids':[7],'logprobs':{'content':[{'token_id':7,'logprob':-1.0}]}}]}
        client._request=request
        return client,requests

if __name__=='__main__':unittest.main()
