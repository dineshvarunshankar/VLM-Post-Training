import unittest


def count_row(prediction, target=2, variant='native'):
    return {'prediction': prediction, 'reference': {'variant': variant},
            'reward_context': {'task': 'counting', 'target': {'count': target},
                               'schema': {'type': 'object', 'properties': {'count': {'type': 'integer', 'minimum': 0}},
                                          'required': ['count'], 'additionalProperties': False}}}


class AuxiliaryEvaluationTests(unittest.TestCase):
    def test_invalid_answers_remain_in_denominator(self):
        from scripts.train.shared.auxiliary_evaluation import score_auxiliary
        report = score_auxiliary([count_row('{"count":2}'), count_row('not json')], {})
        score = report['tasks']['counting']
        self.assertEqual(score['examples'], 2)
        self.assertEqual(score['invalid_rate'], .5)
        self.assertEqual(score['score'], .5)
        self.assertEqual(score['components']['count'], .5)
        self.assertNotIn('macro_f1', report)

    def test_sources_are_reported_separately(self):
        from scripts.train.shared.auxiliary_evaluation import score_auxiliary
        report = score_auxiliary([count_row('{"count":2}'), count_row('{"count":0}', variant='stabilized')], {})
        self.assertEqual(report['sources']['native']['counting']['score'], 1)
        self.assertEqual(report['sources']['stabilized']['counting']['score'], 0)
        self.assertEqual(report['tasks']['counting']['score'], .5)

    def test_fixed_selection_excludes_medical_rows_and_does_not_mutate_plan(self):
        from scripts.train.shared.auxiliary_evaluation import selected_auxiliary
        pool = [{'task':'medical_vqa'}, {'task':'counting'}, {'task':'counting'}]
        plan = {'fixed':[0], 'groups':[{'indices':[1,2], 'values':[None,None], 'count':1,
                                     'balanced':False, 'replace':False}]}
        a = selected_auxiliary(pool, plan, 42)
        self.assertEqual(a, selected_auxiliary(pool, plan, 42))
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]['task'], 'counting')
        self.assertEqual(plan['fixed'], [0])

    def test_invalid_spatial_prediction_is_not_excluded_from_spatial_score(self):
        from scripts.train.shared.auxiliary_evaluation import score_auxiliary
        context={'task':'subject_localization','target':{'point':[10,10]},'subject_box':[0,0,20,20],
                 'schema':{'type':'object','properties':{'point':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}},'required':['point'],'additionalProperties':False}}
        rows=[{'prediction':p,'reference':{'variant':'native'},'reward_context':context}
              for p in ('{"point":[10,10]}','{}')]
        report=score_auxiliary(rows,{})
        self.assertEqual(report['tasks']['subject_localization']['components']['spatial'],.5)

    def test_evaluator_keeps_medical_metric_separate_from_failed_auxiliary(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        import torch
        from scripts.train.shared.evaluation import MedicalVLLMEvaluator
        definition={'subcategories':{'observables':{'cues':[{'name':'cue'}], 'selection_rule':'multi_select_or_none'}}}
        medical={'answer':'{"cue":1}','reference':{'category':'medical','variant':'native'}}
        auxiliary=count_row('{"count":0}')
        auxiliary['answer']='{"count":2}'
        auxiliary['task']='counting'
        schemas=[]
        def generate(*args, **kwargs):
            schemas.append(kwargs['schemas'][0])
            return {'completion_ids':[[len(schemas)-1]]}
        client=NS(generate=generate)
        processor=NS(tokenizer=NS(decode=lambda ids,**kw: ['{"cue":1}','{"count":0}'][ids[0]]))
        generation=dict(temperature=0,top_p=1,top_k=0,repetition_penalty=1,min_p=0)
        evaluator=MedicalVLLMEvaluator(client,{},processor,{'categories':{'medical':definition}},generation,'test',
                                     experiment={},medical_templates={})
        evaluator.auxiliary=NS(scoring={},prepare=lambda dataset,split: ([auxiliary], {'evaluation_examples':1}))
        with tempfile.TemporaryDirectory() as tmp:
            trainer=NS(vllm_generation=NS(sync_weights=lambda:None),state=NS(global_step=2),
                       accelerator=NS(wait_for_everyone=lambda:None,process_index=0,num_processes=1),
                       video_collator=NS(max_length=4096),args=NS(seed=42,output_dir=tmp),
                       is_world_process_zero=lambda:True,log=lambda m:None)
            with patch('scripts.train.shared.evaluation.prepare_video_prompt',return_value={'attention_mask':torch.ones((1,2))}), \
                 patch('scripts.train.shared.evaluation.encode_video_features',return_value={}):
                metrics=evaluator(trainer,[medical],'test')
            report=json.loads((Path(tmp)/'test-step-2.json').read_text())
        self.assertEqual(metrics['test_macro_f1'],1)
        self.assertEqual(metrics['test_examples'],1)
        self.assertEqual(metrics['test_aux/counting/score'],0)
        self.assertEqual(report['auxiliary']['tasks']['counting']['examples'],1)
        self.assertEqual(schemas[0]['required'],['cue'])
        self.assertEqual(schemas[1]['required'],['count'])

    def test_counting_uses_complete_human_visibility_at_each_timestamp(self):
        from scripts.train.shared.grounding import counting_targets
        meta = {'fps': 10, 'n_frames': 10, 'resolution': {'width': 100, 'height': 100},
                'frame_convention': {'first_frame': 1, 'time_s': '(frame - 1) / fps'}}
        counting = {'subject_type': 'casualty_human_actor',
                    'people': [{'id': 1, 'role': 'subject'}, {'id': 2, 'role': 'other'}]}
        def person(identifier, point, by='human', occluded=False):
            return dict(part=f'person:{identifier}', point=point, by=by, occluded=occluded)
        points = {'frames': [
            {'frame': 1, 'parts': [person(1, [20, 20]), person(2, [40, 40])]},
            {'frame': 2, 'parts': [person(1, [20, 20]), person(2, None, occluded=True)]},
            {'frame': 3, 'parts': [person(1, [20, 20]), person(2, [40, 40], by='tracker')]},
            {'frame': 4, 'parts': [person(1, [20, 20])]},
            {'frame': 5, 'parts': [person(1, [20, 20]), person(2, [110, 40])]},
        ]}
        self.assertEqual(counting_targets(counting, points, meta), [
            {'frame': 0, 'timestamp': '00:00', 'count': 2},
            {'frame': 1, 'timestamp': '00:00.1', 'count': 1},
            {'frame': 4, 'timestamp': '00:00.4', 'count': 1},
        ])
