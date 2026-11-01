"""Offline contracts for all-task prompt previews and evidence identity."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import Review, TASKS

SETTINGS = dict(model='gemini/review', video_fps='source', media_resolution='high', reasoning_effort='high', max_tokens=8192)

class PreviewTests(unittest.TestCase):
    def test_saved_response_checks_do_not_certify_visual_truth(self):
        from preview import response_checks
        prepared = {'prompt': {'response_schema': {
            'type': 'object', 'properties': {'answer': {'type': 'string'}},
            'required': ['answer'], 'additionalProperties': False}}}
        checks = response_checks({'answer': 'An unsupported claim.'}, prepared)
        self.assertEqual(checks['response_schema'], 'passed')
        self.assertEqual(checks['visual_support'], 'not_checked')
        self.assertEqual(checks['cross_task_consistency'], 'not_checked')
        self.assertEqual(response_checks({'answer': 123}, prepared)['response_schema'], 'failed')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for variant, dataset, width in [('native','06-native_media_dataset',200), ('stabilized','07-stabilized_media_dataset',100)]:
            clip=self.root/dataset/'example'; ann=clip/'all_annotations'; ann.mkdir(parents=True)
            meta=dict(clip='example', video_path='example.mp4', modality='RGB', annotation_status='completed', fps=10, n_frames=50, duration_s=5, resolution=dict(width=width,height=100), time_of_day='day', scene_description=dict(text='The subject is visible.'), frame_convention=dict(first_frame=1,time_s='(frame - 1) / fps',coordinates='absolute pixels, origin top-left'))
            cues={name:dict(answer=name=='active_bleeding_observed',answered=True,temporal=name=='active_bleeding_observed',evidence_min=1 if name=='active_bleeding_observed' else 0,evidence_slots=None,evidence_side_options=None,evidence_points=[],time_spans=[]) for name in ['active_bleeding_observed','blood_pooling_observed','extensive_blood_on_body_observed','hemorrhage_assessed_absent','hemorrhage_not_assessable']}
            cues['active_bleeding_observed'].update(evidence_points=[dict(frame=24,time_s=2.35,point=[50,25],slot='left arm')],time_spans=[dict(start_s=2.1,end_s=2.8)])
            category=dict(observables=dict(cues={k:v for k,v in cues.items() if 'observed' in k}),assessment=dict(cues={k:v for k,v in cues.items() if 'observed' not in k}),narration=dict(transcript=dict(text='A localized observation.')))
            values={'meta':meta,'camera_motion':dict(view_type='aerial_oblique'),'counting':dict(person_count=1,subject_count=1,subject_type='casualty_human_actor',other_people_count=0,robot_count=0),'labels':dict(categories={'SEVERE HEMORRHAGE':category}),'points':dict(frames=[dict(frame=24,parts=[dict(part='evidence:active_bleeding_observed#0',point=[50,25],slot='left arm',occluded=False,by='cotracker')])])}
            for name,value in values.items(): (ann/(name+'.json')).write_text(json.dumps(value))
            video=clip/'example.mp4'; video.write_bytes(b'fixture video')
            sidecar=clip/'example.json'; sidecar.write_text('{}')
            digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            (clip/'provenance.json').write_text(json.dumps(dict(clip='example',variant=variant,output_sha256=digest(video),sidecar_sha256=digest(sidecar),annotation_sha256={p.name:digest(p) for p in ann.glob('*.json')})))
            (clip.parent/'dataset_manifest.json').write_text(json.dumps(dict(schema_version=1,variant=variant,clips=[dict(clip_id='example',clip='example',level='full',provenance_sha256=digest(clip/'provenance.json'))])))
        self.review=Review(self.root,SETTINGS)

    def test_native_preview_matches_generation_request(self):
        from generate import prepare, request_payload
        from preview import PROMPT
        prepared = prepare(self.root/'06-native_media_dataset/example', PROMPT, SETTINGS)
        expected = request_payload(prepared, '<uploaded_video_uri>')
        result = self.review.preview('example')
        self.assertEqual(result['request'], expected)
        self.assertEqual(result['user_prompt'], expected['messages'][1]['content'][1]['text'])
        self.assertEqual(result['system_prompt'], expected['messages'][0]['content'])
        self.assertEqual(result['schema'], expected['response_format']['json_schema']['schema'])

    def test_all_tasks_use_generation_inputs(self):
        from unittest.mock import patch
        for task in TASKS:
            with self.subTest(task=task), patch('medical.frame_times',return_value=[i/10 for i in range(50)]):
                self.assertTrue(self.review.template(task)['production_request'])
                self.assertEqual(len(self.review.clips(task)),1)
                result=self.review.preview('example', task)
                self.assertNotIn('{metadata_json}',result['user_prompt'])
                if task.startswith('stabilized'):
                    self.assertNotIn('recorded_clip_totals',result['user_prompt'])
                if 'medical' in task:
                    self.assertFalse(result['schema']['properties']['answer']['additionalProperties'])
                    self.assertEqual(result['evidence_frames'][0]['time'],'00:02.3')
                    self.assertTrue(any(p['type']=='image_url' for p in result['request']['messages'][1]['content']))

    def test_medical_payload_contains_only_model_fields(self):
        from unittest.mock import patch
        from medical import medical_inputs, CATALOG, TEMPLATES
        from generate import prepare, request_payload
        from preview import PROMPTS
        from scripts.train.shared.data import read_yaml, category_prompt_values
        from evidence_media import evidence_images
        from types import SimpleNamespace
        for variant in ('native','stabilized'):
            task=variant+'_medical_spatiotemporal_vqa'
            clip=self.root/('06-native_media_dataset' if variant=='native' else '07-stabilized_media_dataset')/'example'
            with self.subTest(variant=variant), patch('medical.frame_times',return_value=[i/10 for i in range(50)]):
                p=prepare(clip,PROMPTS/(task+'.yaml'),SETTINGS,'severe_hemorrhage')
                request=request_payload(p,'<video>')
                for field in ('frame_index','answered','by'):
                    self.assertNotIn('"'+field+'"', '\n'.join(part.get('text','') for part in request['messages'][1]['content']))
                self.assertNotIn('Return the JSON object with these label keys',p['text'])
                definition=read_yaml(CATALOG)['categories']['severe_hemorrhage']
                values=category_prompt_values(definition,read_yaml(TEMPLATES),'aerial_oblique')
                self.assertEqual(p['text'].count(values['definitions']),1)
                self.assertEqual(p['text'].count(values['selection_rules']),1)
                self.assertEqual(p['evidence_frames'][0]['frame_index'],23)
                self.assertEqual(p['evidence']['active_bleeding_observed']['tracks'][0]['by'],'cotracker')
                with patch('evidence_media.subprocess.run',return_value=SimpleNamespace(stdout=b'\x89PNGfake')) as extract:
                    parts=evidence_images(p)
                self.assertIn('23',str(extract.call_args))
                self.assertNotIn('frame_index',parts[0]['text'])
                self.assertIn('00:02.3',parts[0]['text'])

    def test_only_mandatory_grounding_is_supplied(self):
        from unittest.mock import patch
        from medical import medical_inputs
        from preview import PROMPTS
        import yaml
        for variant in ('native', 'stabilized'):
            clip = self.root / ('06-native_media_dataset' if variant == 'native' else '07-stabilized_media_dataset') / 'example'
            labels_path = clip / 'all_annotations/labels.json'
            labels = json.loads(labels_path.read_text())
            cue = labels['categories']['SEVERE HEMORRHAGE']['observables']['cues']['active_bleeding_observed']
            meta = json.loads((clip / 'all_annotations/meta.json').read_text())
            schema = yaml.safe_load((PROMPTS / (variant + '_medical_spatiotemporal_vqa.yaml')).read_text())['response_schema']
            for selected, minimum, temporal in ((True, 0, False), (True, 1, False), (True, 0, True), (True, 1, True), (False, 1, True)):
                with self.subTest(variant=variant, selected=selected, minimum=minimum, temporal=temporal):
                    cue.update(answer=selected, evidence_min=minimum, temporal=temporal)
                    labels['categories']['SEVERE HEMORRHAGE']['assessment']['cues']['hemorrhage_assessed_absent']['answer'] = not selected
                    labels_path.write_text(json.dumps(labels))
                    with patch('medical.frame_times', return_value=[i/10 for i in range(50)]):
                        payload, _, prepared = medical_inputs(clip, meta, 'aerial_oblique', 'severe_hemorrhage', variant, schema)
                    evidence = json.loads(payload['evidence_annotations_json'])['active_bleeding_observed']
                    self.assertEqual(bool(evidence['points']), selected and minimum > 0)
                    self.assertEqual(bool(evidence['tracks']), selected and minimum > 0)
                    self.assertEqual(bool(evidence['spans']), selected and temporal)
                    self.assertEqual(bool(prepared['evidence_frames']), selected and minimum > 0)

    def test_optional_grounding_is_rejected(self):
        from medical import validate_medical
        prepared = dict(expected_answer={'cue': 1},
                        requirements={'cue': dict(evidence_min=0, temporal=False)},
                        evidence={'cue': dict(points=[dict(point=[100,200], time='00:01')], tracks=[],
                                              spans=[dict(start='00:01', end='00:02')])})
        validate_medical(dict(answer={'cue': 1}, reasoning_process='An assessment.'), prepared)
        for markup in ('<point>[100,200]</point><time>00:01</time>', '<span>00:01–00:02</span>'):
            with self.subTest(markup=markup), self.assertRaises(ValueError):
                validate_medical(dict(answer={'cue': 1}, reasoning_process='<evidence><cue>cue</cue>' + markup + '</evidence>'), prepared)

    def test_scene_context_uses_labels_definitions_and_positive_spans(self):
        from medical import scene_medical_context
        for variant in ('native', 'stabilized'):
            clip = self.root / ('06-native_media_dataset' if variant == 'native' else '07-stabilized_media_dataset') / 'example'
            meta = json.loads((clip / 'all_annotations/meta.json').read_text())
            context, excluded = scene_medical_context(clip, meta, variant)
            category = context['severe_hemorrhage']
            records = {row['label']: row for row in category['labels']}
            self.assertEqual(records['active_bleeding_observed']['answer'], 1)
            self.assertEqual(records['active_bleeding_observed']['spans'], [{'start': '00:02.1', 'end': '00:02.8'}])
            self.assertTrue(records['active_bleeding_observed']['definition'])
            self.assertEqual(records['blood_pooling_observed']['answer'], 0)
            self.assertNotIn('spans', records['blood_pooling_observed'])
            self.assertTrue(excluded)
            self.assertNotIn('points', json.dumps(context))
            self.assertNotIn('transcript', json.dumps(context))
            result = self.review.preview('example', variant + '_general_scene_description')
            self.assertIn('HUMAN MEDICAL ANNOTATIONS FOR CONSISTENCY', result['user_prompt'])
            self.assertIn('active_bleeding_observed', result['user_prompt'])
            self.assertIn('catalog_sha256', result['inputs'])

    def test_medical_coordinates_are_integers_and_tracks_cannot_be_cited(self):
        from unittest.mock import patch
        from medical import medical_inputs, validate_medical
        from preview import PROMPTS
        import yaml
        for variant in ('native', 'stabilized'):
            clip = self.root / ('06-native_media_dataset' if variant == 'native' else '07-stabilized_media_dataset') / 'example'
            labels_path = clip / 'all_annotations/labels.json'
            labels = json.loads(labels_path.read_text())
            cue = labels['categories']['SEVERE HEMORRHAGE']['observables']['cues']['active_bleeding_observed']
            cue['evidence_points'][0]['point'] = [50.15, 25.15]
            labels_path.write_text(json.dumps(labels))
            meta = json.loads((clip / 'all_annotations/meta.json').read_text())
            schema = yaml.safe_load((PROMPTS / (variant + '_medical_spatiotemporal_vqa.yaml')).read_text())['response_schema']
            with patch('medical.frame_times', return_value=[i/10 for i in range(50)]):
                _, _, prepared = medical_inputs(clip, meta, 'aerial_oblique', 'severe_hemorrhage', variant, schema)
            point = prepared['evidence']['active_bleeding_observed']['points'][0]['point']
            self.assertEqual(point, [251 if variant == 'native' else 502, 252])
            self.assertTrue(all(type(value) is int for value in point))
            cue_name = 'active_bleeding_observed'
            def ref(xy):
                return '<evidence><cue>' + cue_name + '</cue><slot>left arm</slot><point>' + json.dumps(xy) + '</point><time>00:02.3</time></evidence>'
            rationale = ref(point) + '<evidence><cue>' + cue_name + '</cue><span>00:02.1–00:02.8</span></evidence>'
            response = dict(answer=prepared['expected_answer'], reasoning_process=rationale)
            validate_medical(response, prepared)
            with self.assertRaises(ValueError):
                validate_medical(dict(response, reasoning_process=rationale.replace(json.dumps(point), json.dumps([float(v) for v in point]))), prepared)
            response['reasoning_process'] += ref(prepared['evidence'][cue_name]['tracks'][0]['point'])
            with self.assertRaises(ValueError):
                validate_medical(response, prepared)

    def test_gateway_uses_inline_video_and_high_reasoning(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from generate import prepare, request_payload, generate
        from preview import PROMPT
        settings=dict(SETTINGS,model='gemini/gemini-3.8-flash',api_base='https://gateway.example/v1')
        prepared=prepare(self.root/'06-native_media_dataset/example',PROMPT,settings)
        payload=request_payload(prepared,'<video>')
        self.assertEqual(payload['model'],'litellm_proxy/gemini/gemini-3.8-flash')
        self.assertEqual(payload['api_base'],settings['api_base'])
        self.assertEqual(payload['reasoning_effort'],'high')
        self.assertIn('file_data',payload['messages'][1]['content'][0]['file'])
        with patch('generate.upload_video') as upload, patch('litellm.completion') as api, patch.dict('os.environ',{'GEMINI_API_KEY':'test-only'}):
            api.return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"answer":"Scene."}'))],usage=None,id='test')
            generate(prepared,self.root/'cache')
            upload.assert_not_called()
            sent=api.call_args.kwargs
            self.assertTrue(sent['messages'][1]['content'][0]['file']['file_data'].startswith('data:video/mp4;base64,'))
            self.assertEqual(sent['api_key'],'test-only')
        self.assertNotIn('api_key',payload)

    def test_generation_preserves_gt_and_previous_result(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from generate import prepare, process_clip
        from preview import PROMPTS
        task='native_medical_spatiotemporal_vqa'
        clip=self.root/'06-native_media_dataset/example'
        prompt=PROMPTS/(task+'.yaml')
        with patch('medical.frame_times',return_value=[i/10 for i in range(50)]):
            prepared=prepare(clip,prompt,SETTINGS,'severe_hemorrhage')
            answer=prepared['expected_answer']
            rationale=('Visible finding <evidence><cue>active_bleeding_observed</cue><slot>left arm</slot>'
                       '<point>[250,250]</point><time>00:02.3</time></evidence> across '
                       '<evidence><cue>active_bleeding_observed</cue><span>00:02.1–00:02.8</span></evidence>.')
            def reply(answer):
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(
                    content=json.dumps(dict(answer=answer,reasoning_process=rationale))))],usage=None,id='offline-test')
            with patch('generate.upload_video',return_value='mock-uri'), patch('generate.evidence_images',return_value=[]), patch('litellm.completion',return_value=reply(answer)) as api:
                _,report=process_clip(clip,prompt,SETTINGS,self.root/'cache',write=True,inspect=False,category='severe_hemorrhage')
                self.assertEqual(report['status'],'generated')
                saved=Path(report['output']); original=saved.read_bytes()
                self.assertIn('text_statistics', json.loads(original)['review'])
                self.assertIn('review', report)
                _,again=process_clip(clip,prompt,SETTINGS,self.root/'cache',write=True,inspect=False,category='severe_hemorrhage')
                self.assertEqual(again['status'],'current');self.assertEqual(api.call_count,1)
                wrong=dict(answer);wrong['active_bleeding_observed']=0;api.return_value=reply(wrong)
                _,failed=process_clip(clip,prompt,dict(SETTINGS,max_tokens=9000),self.root/'cache',write=True,inspect=False,category='severe_hemorrhage')
                self.assertEqual(failed['status'],'error');self.assertEqual(saved.read_bytes(),original)

    def test_focused_preview_limits_clips_and_categories(self):
        from unittest.mock import patch
        focused = Review(self.root, SETTINGS, focus_clips={'native': 'example', 'stabilized': 'example'}, focus_category='severe_hemorrhage')
        self.assertTrue(focused.template()['focused'])
        for task in TASKS:
            self.assertEqual([row['id'] for row in focused.clips(task)], ['example'])
        with self.assertRaises(ValueError):
            focused.clip('another')
        with patch('medical.frame_times', return_value=[i/10 for i in range(50)]):
            result = focused.preview('example', 'native_medical_spatiotemporal_vqa')
            self.assertEqual(result['categories'], ['severe_hemorrhage'])
            self.assertEqual(result['category'], 'severe_hemorrhage')
            with self.assertRaises(ValueError):
                focused.preview('example', 'native_medical_spatiotemporal_vqa', 'respiratory_distress')

    def test_evidence_preview_uses_exact_annotated_frame(self):
        from unittest.mock import patch
        with patch('medical.frame_times', return_value=[i/10 for i in range(50)]):
            result = self.review.preview('example', 'native_medical_spatiotemporal_vqa', 'severe_hemorrhage')
            point = result['human_evidence']['active_bleeding_observed']['points'][0]
            self.assertEqual(point['frame_index'], 23)
            self.assertEqual(point['point'], [250, 250])
            with patch('preview.evidence_images', return_value=[{}, {'image_url': {'url': 'data:image/png;base64,iVBORw=='}}]) as render:
                data = self.review.evidence_frame('example', 'native_medical_spatiotemporal_vqa', 'severe_hemorrhage', 23)
                self.assertEqual(data, b'\x89PNG')
                self.assertEqual(render.call_args.args[0]['evidence_frames'][0]['frame_index'], 23)
            with self.assertRaises(ValueError):
                self.review.evidence_frame('example', 'native_medical_spatiotemporal_vqa', 'severe_hemorrhage', 22)
            with self.assertRaises(ValueError):
                self.review.evidence_frame('example', 'native_general_scene_description', '', 23)

    def test_media_paths_cannot_escape_dataset(self):
        with self.assertRaises(ValueError): self.review.video('../example')

if __name__=='__main__': unittest.main()
