"""Task reuse tracks model inputs and still enforces source integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.train.tests.test_cot_data import fixture, COT, CATALOG
from scripts.train.shared.data import sha256, read_yaml
from scripts.train.shared.cot_data import validate_target, SourceManifest
import sys
sys.path.insert(0,str(COT))
from scripts.cot import generate as engine

def republish(folder):
    p=folder/'provenance.json'; data=json.loads(p.read_text())
    data['annotation_sha256']={p.name:sha256(p) for p in (folder/'all_annotations').glob('*.json')}
    data['sidecar_sha256']=sha256(folder/(folder.name+'.json'))
    data['output_sha256']=sha256(folder/(folder.name+'.mp4'))
    p.write_text(json.dumps(data))
    p=folder.parent/'dataset_manifest.json'; data=json.loads(p.read_text())
    for row in data['clips']:
        if row['clip_id']==folder.name:row['provenance_sha256']=sha256(folder/'provenance.json')
    p.write_text(json.dumps(data))

class TaskFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.folder,self.rows=fixture(self.root)
        self.catalog=read_yaml(CATALOG)
    def prepare(self,category='severe_hemorrhage'):
        task='native_'+('medical_spatiotemporal_vqa' if category else 'general_scene_description')
        return engine.prepare(self.folder,COT/'yaml/RGB'/(task+'.yaml'),{'video_fps':2},category)
    def target(self,category='severe_hemorrhage'):
        return self.folder/'GT'/('medical_spatiotemporal_vqa/'+category+'.json' if category else 'general_scene_description.json')
    def edit(self,name,fn,publish=True):
        path=self.folder/'all_annotations'/name; value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))
        if publish:republish(self.folder)
    def narration(self,category):
        name=self.catalog['categories'][category]['name']
        self.edit('labels.json',lambda x:x['categories'][name].update(narration={'transcript':{'text':'Updated category narration'}}))
    def test_unrelated_category_reuses_in_generator_and_training(self):
        self.narration('medic_presence')
        self.assertTrue(engine.is_current(self.target(),self.prepare()))
        self.assertFalse(engine.is_current(self.target('medic_presence'),self.prepare('medic_presence')))
        row=dict(self.rows[0],task='medical_vqa')
        validate_target(self.target(),row,self.catalog,{'cot':{'accepted_implementation_fingerprints':[]}},SourceManifest(self.root))
    def test_bookkeeping_only_reuses(self):
        self.edit('meta.json',lambda x:x.update(last_reviewed='today'))
        self.assertTrue(engine.is_current(self.target(),self.prepare()))
    def test_relevant_transcript_invalidates(self):
        self.narration('severe_hemorrhage')
        self.assertFalse(engine.is_current(self.target(),self.prepare()))
    def test_unpublished_edit_still_blocks(self):
        self.edit('meta.json',lambda x:x.update(last_reviewed='today'),publish=False)
        with self.assertRaises(ValueError):self.prepare()
    def test_viewpoint_invalidates(self):
        self.edit('camera_motion.json',lambda x:x.update(view_type='oblique'))
        self.assertFalse(engine.is_current(self.target(),self.prepare()))
    def test_source_narration_invalidates_scene(self):
        self.edit('meta.json',lambda x:x.update(scene_description={'text':'Different visible scene'}))
        self.assertFalse(engine.is_current(self.target(None),self.prepare(None)))

    def test_label_change_invalidates_category_and_scene_but_not_other_category(self):
        def change(data):
            groups=data['categories'][self.catalog['categories']['medic_presence']['name']]
            for group in groups.values():
                for name,cue in group.get('cues',{}).items():
                    cue['answer']=name=='medic_assessed_absent'
        self.edit('labels.json',change)
        self.assertTrue(engine.is_current(self.target(),self.prepare()))
        self.assertFalse(engine.is_current(self.target('medic_presence'),self.prepare('medic_presence')))
        self.assertFalse(engine.is_current(self.target(None),self.prepare(None)))

    def test_new_input_dependency_fails_closed(self):
        from copy import deepcopy
        from scripts.cot.task_identity import same_task_inputs
        original=self.prepare()['inputs']
        for key in ('video_sha256','prompt_sha256','settings','libraries','implementation',
                    'task_input_sha256','assembled_input_sha256','video_fps_policy','future_dependency'):
            changed=deepcopy(original);changed[key]='changed'
            with self.subTest(key=key):self.assertFalse(same_task_inputs(original,changed))
        changed=deepcopy(original);changed['task_identity_version']=999
        self.assertFalse(same_task_inputs(original,changed))

    def test_exact_frame_and_evidence_changes_invalidate_identity(self):
        from copy import deepcopy
        from scripts.cot.task_identity import task_input_hash
        prompt={'name':'medical','system_prompt':'Assess video','user_prompt':'{question}'}
        values={'question':'Assess bleeding'}; schema={'type':'object'}
        evidence={'evidence_frames':[{'frame_index':1,'time':'00:00.001','cues':['bleeding']}],
                  'expected_answer':{'bleeding':1},'requirements':{'bleeding':{'evidence_min':1}},
                  'evidence':{'bleeding':{'points':[{'point':[500,500],'time':'00:00.001','slot':'head'}],
                                          'spans':['00:00–00:01']}}}
        original=task_input_hash(prompt,values,schema,evidence)
        changes=[lambda x:x['evidence_frames'][0].update(frame_index=2),
                 lambda x:x['evidence_frames'][0].update(time='00:01'),
                 lambda x:x['evidence']['bleeding']['points'][0].update(point=[501,500]),
                 lambda x:x['evidence']['bleeding']['points'][0].update(slot='torso'),
                 lambda x:x['evidence']['bleeding'].update(spans=['00:00–00:02']),
                 lambda x:x['requirements']['bleeding'].update(evidence_min=2)]
        for change in changes:
            modified=deepcopy(evidence);change(modified)
            self.assertNotEqual(original,task_input_hash(prompt,values,schema,modified))
        self.assertNotEqual(original,task_input_hash(prompt,values,{'type':'string'},evidence))
        self.assertNotEqual(original,task_input_hash(dict(prompt,system_prompt='Different instruction'),values,schema,evidence))

    def test_legacy_upgrade_requires_all_old_dependencies(self):
        from copy import deepcopy
        from scripts.cot.task_identity import legacy_current_inputs
        current=self.prepare()['inputs'];previous=legacy_current_inputs(current)
        self.assertIsNotNone(previous)
        self.assertTrue(engine.compatible_inputs(previous,current))
        for key in ('annotation_sha256','video_sha256','provenance_sha256','assembled_input_sha256','implementation'):
            modified=deepcopy(previous);modified[key]={'changed':'different'}
            self.assertFalse(engine.compatible_inputs(modified,current))

    def test_missing_duration_and_transcript_match_training_reconstruction(self):
        self.edit('meta.json',lambda x:x.update(duration_s=None,scene_description={'text':None}))
        prepared=self.prepare();path=self.target();saved=json.loads(path.read_text())
        saved['generation'].update(inputs=prepared['inputs'],signature=prepared['signature'])
        path.write_text(json.dumps(saved))
        validate_target(path,dict(self.rows[0],task='medical_vqa'),self.catalog,
                        {'cot':{'accepted_implementation_fingerprints':[]}},SourceManifest(self.root))

if __name__=='__main__':unittest.main()
