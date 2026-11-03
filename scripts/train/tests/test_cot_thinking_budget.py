"""Medical reasoning budgets reach the native vLLM sampler, with safe defaults."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.train.shared.run_config import compile_run
from scripts.train.tests.test_cot_integration import run_config
from scripts.train.tests.test_cot_data import REPO
from scripts.train.shared.data import read_yaml
from scripts.train.rl.rollout.client import VideoRolloutClient


class ThinkingBudgetTests(unittest.TestCase):
    def test_all_drafts_resolve_3064_within_4096_without_changing_training_context(self):
        for model in ('qwen3.5-9b','qwen3.8-27b','gemma4-12b-it','gemma4-31b-it'):
            raw=read_yaml(REPO/'scripts/train/configs/cot'/(model+'.yaml'))
            raw['paths'].update(media_root='/explicit/media',split_dir='/explicit/splits')
            result=compile_run(raw)['experiment']
            self.assertEqual(result['cot'].get('thinking_token_budget'),3064)
            self.assertEqual(result['cot']['medical_max_new_tokens'],4096)
            self.assertEqual(result['max_length'],262144)

    def test_config_rejects_invalid_budget_or_insufficient_final_reserve(self):
        raw=run_config();raw['shared']['cot']['thinking_token_budget']=3064
        try: compile_run(raw)
        except ValueError as error:self.fail(f'Valid reasoning budget rejected: {error}')
        for budget in (-1,True,3064.0,None,3065,4096):
            changed=deepcopy(raw);changed['shared']['cot']['thinking_token_budget']=budget
            with self.assertRaisesRegex(ValueError,'thinking_token_budget|reserve'):
                compile_run(changed)

    def test_evaluator_sends_budget_only_for_medical(self):
        from scripts.train.shared.cot_evaluation import CoTVLLMEvaluator
        experiment=compile_run(run_config())['experiment']
        experiment['cot']['thinking_token_budget']=3064
        requests=[]
        class Client:
            def generate(self,packets,**kwargs):
                requests.append(kwargs)
                return {'completion_ids':[[1]],'finish_reasons':['stop']}
        processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:'Visible evidence.</think>{"injury":1}<|im_end|>'))
        evaluator=CoTVLLMEvaluator(Client(),{},processor,{}, {'do_sample':False,'temperature':0,'top_p':1,'top_k':0,'min_p':0,'repetition_penalty':1},'run',model_family='qwen3_5',experiment=experiment)
        for task in ('medical_vqa','scene_description','camera_view'):
            schema=None if task=='scene_description' else {'type':'object','properties':{'injury':{'type':'integer','enum':[0,1]}},'required':['injury']}
            evaluator.predict_encoded({'packet':{},'task':task},'version',42,schema)
        self.assertEqual(requests[0].get('thinking_token_budget'),3064)
        self.assertEqual(requests[0].get('reasoning_parser'),'qwen3')
        self.assertEqual(requests[0]['max_tokens'],4096)
        for request in requests[1:]:self.assertNotIn('thinking_token_budget',request)

    def test_client_places_budget_in_sampling_params_and_rejects_unready_parser(self):
        client=VideoRolloutClient('http://unused','model');client._checked=True
        packet={'token_ids':[1000],'features':{'mm_placeholders':{}}}
        response={'prompt_token_ids':[1000],'mm_placeholders':{},'choices':[{'index':0,'finish_reason':'stop','token_ids':[7],'logprobs':{'content':[{'token_id':7,'logprob':-1.0}]}}]}
        requests=[]
        def request(method,path,**kwargs):
            requests.append((path,kwargs))
            if path=='/server_info':
                return {'vllm_config':{'reasoning_config':{'reasoning_parser':'qwen3','_enabled':True,'_reasoning_start_token_ids':[1],'_reasoning_end_token_ids':[2]}}}
            return response
        try:
            with patch.object(client,'_request',side_effect=request):
                client.generate([packet],policy_version='v',temperature=0,top_p=1,top_k=0,repetition_penalty=1,max_tokens=4096,thinking_token_budget=3064,reasoning_parser='qwen3')
                client.generate([packet],policy_version='v',temperature=0,top_p=1,top_k=0,repetition_penalty=1,max_tokens=1024)
        except TypeError as error:self.fail(f'Client does not support the sampler budget: {error}')
        bodies=[item['json'] for path,item in requests if path=='/inference/v1/generate']
        self.assertEqual(bodies[0]['sampling_params']['thinking_token_budget'],3064)
        self.assertNotIn('thinking_token_budget',bodies[0])
        self.assertNotIn('thinking_token_budget',bodies[1]['sampling_params'])
        bad=VideoRolloutClient('http://unused','model');bad._checked=True
        with patch.object(bad,'_request',return_value={'vllm_config':{'reasoning_config':None}}):
            with self.assertRaisesRegex(ValueError,'reasoning-parser qwen3'):
                bad.check_thinking_budget_compatibility('qwen3')

    def test_readiness_rejects_marker_overrides_that_disable_native_budget_tracking(self):
        for parser,start,end in [('qwen3','<think>','</think>'),('gemma4','<|channel>','<channel|>')]:
            for bad_field in ('reasoning_start_str','reasoning_end_str'):
                reasoning={'reasoning_parser':parser,'_enabled':True,'_reasoning_start_token_ids':[1],
                           '_reasoning_end_token_ids':[2],bad_field:'OTHER'}
                client=VideoRolloutClient('http://unused','model')
                with patch.object(client,'_request',return_value={'vllm_config':{'reasoning_config':reasoning}}):
                    with self.assertRaisesRegex(ValueError,'reasoning-parser'):
                        client.check_thinking_budget_compatibility(parser)
            reasoning.update(reasoning_start_str=start,reasoning_end_str=end)
            client=VideoRolloutClient('http://unused','model')
            with patch.object(client,'_request',return_value={'vllm_config':{'reasoning_config':reasoning}}):
                client.check_thinking_budget_compatibility(parser)

    def test_saved_policy_and_resume_keep_resolved_budget(self):
        import json,tempfile
        from scripts.train.shared.runtime import save_run_metadata
        from scripts.train.tests.test_evaluation_decoding import GREEDY
        config=compile_run(run_config())['experiment']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('catalog.yaml','prompts.yaml','train.jsonl'):(root/name).write_text('')
            args=SimpleNamespace(model='qwen3.5-9b',run_location={'answer_format':'cot'},resume=None,
                catalog=root/'catalog.yaml',prompts=root/'prompts.yaml',split_dir=root)
            output=root/'output'
            with patch.dict('os.environ',{'TRIAGE_RUN_CONFIG':''}):
                save_run_metadata(args,output,{'generation':GREEDY},config,{}, {}, {'train':[]})
                policy=json.loads((output/'cot_generation_policy.json').read_text())
                self.assertEqual(policy['medical_vqa']['thinking_token_budget'],3064)
                self.assertEqual(policy['medical_vqa']['max_new_tokens'],4096)
                self.assertFalse(policy['scene_description']['thinking'])
                args.resume=output/'checkpoints/checkpoint-1'
                try:save_run_metadata(args,output,{'generation':GREEDY},config,{}, {}, {'train':[]})
                except ValueError as error:self.fail(f'Unchanged resolved budget failed resume comparison: {error}')

    def test_server_args_are_opt_in_and_model_specific(self):
        try:
            from scripts.train.shared.cot_budget import server_arguments
        except ImportError:self.fail('CoT server parser resolution is missing')
        for model,parser in [('qwen3.5-9b','qwen3'),('qwen3.8-27b','qwen3'),('gemma4-12b-it','gemma4'),('gemma4-31b-it','gemma4')]:
            self.assertEqual(server_arguments({'answer_format':'cot','model':model}),['--reasoning-parser',parser])
        self.assertEqual(server_arguments({'answer_format':'answer_only','model':'gemma4-31b-it'}),[])


if __name__=='__main__':unittest.main()
