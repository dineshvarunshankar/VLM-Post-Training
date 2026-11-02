"""CoT targets must stay tied to frozen human labels and current source bytes."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from copy import deepcopy

from scripts.train.shared.data import read_yaml, sha256
from scripts.train.shared.run_config import compile_run, load_run

REPO = Path(__file__).resolve().parents[3]
CATALOG = REPO / 'scripts/train/configs/catalogs/RGB/label_catalog.yaml'
COT = REPO / 'scripts/cot'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def fixture(root, *, variant='native', clip_id='abc', targets=True):
    folder = root / ('06-native_media_dataset' if variant == 'native' else '07-stabilized_media_dataset') / clip_id
    ann = folder / 'all_annotations'
    ann.mkdir(parents=True)
    video = folder / (clip_id + '.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=size=64x64:rate=2:duration=1', '-c:v', 'libx264', '-threads', '1', str(video)], check=True)
    meta = dict(clip=clip_id+'_i1s1_C1', video_path=str(video), modality='RGB', annotation_status='completed',
                fps=2, n_frames=2, duration_s=1, time_of_day='day', resolution=dict(width=64, height=64),
                scene_description=dict(text='PRIVATE TRANSCRIPT'), frame_convention=dict(first_frame=1, time_s='(frame - 1) / fps', coordinates='absolute pixels, origin top-left'))
    catalog = read_yaml(CATALOG)
    answers, labels = {}, {'categories': {}}
    for category in ('severe_hemorrhage', 'medic_presence'):
        definition = catalog['categories'][category]
        answer = {cue['name']: int(cue['name'].endswith('_not_assessable')) for group in definition['subcategories'].values() for cue in group['cues']}
        answers[category] = answer
        labels['categories'][definition['name']] = {name: dict(cues={cue['name']:dict(answered=True, answer=bool(answer[cue['name']]), temporal=False, evidence_min=0) for cue in group['cues']}) for name, group in definition['subcategories'].items()}
    sources = {'meta.json': meta, 'labels.json': labels, 'camera_motion.json': {'view_type': 'nadir'}}
    if variant=='native':
        sources['counting.json']={'person_count':1,'subject_count':1,'subject_type':'casualty','other_people_count':0,'robot_count':0}
    for name, value in sources.items():
        (ann / name).write_text(json.dumps(value))
    video.with_suffix('.json').write_text(json.dumps({'clip':meta['clip']}))
    (folder / 'provenance.json').write_text(json.dumps({'clip':meta['clip'],'sidecar_sha256':sha256(video.with_suffix('.json')),'output_sha256':sha256(video), 'annotation_sha256':{name:sha256(ann/name) for name in sources}}))
    (folder / 'GT').mkdir()
    (folder / 'GT/medical_vqa.json').write_text(json.dumps({'categories':answers,'source_hashes':{'labels_sha256':sha256(ann/'labels.json'),'label_catalog_sha256':sha256(CATALOG)}}))
    rows = [dict(video=str(video), system='Assess visible medical cues.', question='Assess ' + cat, answer=json.dumps(answer,separators=(',',':')),
                 reference=dict(variant=variant,clip=clip_id,category=cat), video_sha256=sha256(video), video_metadata={'fps':2,'n_frames':2}) for cat,answer in answers.items()]
    # Use the actual generator's preparation contract for tiny on-disk targets.
    for source,name in [('native','06-native_media_dataset'),('stabilized','07-stabilized_media_dataset')]:
        manifest_path=root/name/'dataset_manifest.json'
        manifest_path.parent.mkdir(exist_ok=True)
        document=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version':1,'variant':source,'clips':[]}
        if source==variant:
            document['clips'].append({'clip_id':clip_id,'clip':meta['clip'],'level':'full','provenance_sha256':sha256(folder/'provenance.json')})
        manifest_path.write_text(json.dumps(document))
    if targets:
        import sys
        sys.path.insert(0,str(COT))
        try:
            generator=importlib.import_module('scripts.cot.generate')
        finally:
            sys.path.remove(str(COT))
        for category in [*answers,None]:
            task=variant+('_medical_spatiotemporal_vqa' if category else '_general_scene_description')
            prepared=generator.prepare(folder,COT/'yaml/RGB'/(task+'.yaml'),{'video_fps':2},category)
            value={'generation':{'task':task,'category':category,'signature':prepared['signature'],'inputs':prepared['inputs']},
                   'response':{'reasoning_process':'The view does not support a reliable assessment.','answer':answers[category]} if category else {'answer':'A person is visible in the scene.'}}
            path=folder/'GT'/('medical_spatiotemporal_vqa/'+category+'.json' if category else 'general_scene_description.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(value))
    return folder, rows


def config():
    c = compile_run(load_run(REPO/'scripts/train/configs/runs/qwen9b_sft_native_stabilized_medical_camera_viewpoint.yaml'))['experiment']
    c['answer_format']='cot'
    c['cot']={'medical_max_new_tokens':4096,'thinking_token_budget':3064,'caption_max_new_tokens':1024,'caption_review_samples':4,'accepted_implementation_fingerprints':[]}
    return c


class CoTDataTests(unittest.TestCase):
    def builder(self):
        try: module=importlib.import_module('scripts.train.shared.cot_data')
        except ModuleNotFoundError: module=None
        self.assertIsNotNone(module, 'CoT target attachment is not implemented')
        return module.build_cot_splits

    def build(self, root, splits):
        return self.builder()(splits, media_root=root, catalog=read_yaml(CATALOG), config=config())

    def test_caption_deduplicated_and_native_camera_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,native=fixture(root)
            _,stable=fixture(root,variant='stabilized',clip_id='def')
            result,report=self.build(root, {'train':native+stable,'validation':[],'test':[]})
            tasks=[r['task'] for r in result['train']]
            self.assertEqual(tasks.count('medical_vqa'),4)
            self.assertEqual(tasks.count('scene_description'),2)
            self.assertEqual(tasks.count('camera_view'),1)
            self.assertTrue(report['release_ready'])
            self.assertTrue(report['manifest']['files'])
            for row in result['train']:
                self.assertNotIn('PRIVATE TRANSCRIPT',row['system']+row['question'])
                self.assertNotIn('reasoning_process',row['question'])

    def test_heldout_gt_available_without_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,train=fixture(root)
            _,held=fixture(root,clip_id='held',targets=False)
            result,report=self.build(root, {'train':train,'validation':held,'test':[]})
            self.assertEqual([r['answer'] for r in result['validation']], [r['answer'] for r in held])
            self.assertTrue(all(r['reasoning'] is None for r in result['validation']))
            self.assertEqual(report['teacher_loss']['validation'],{'expected':2,'covered':0})
            self.assertTrue(report['release_ready'])

    def test_missing_training_target_blocks_with_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); _,rows=fixture(root,targets=False)
            result,report=self.build(root,{'train':rows,'validation':[],'test':[]})
            self.assertFalse(report['release_ready'])
            self.assertEqual(report['coverage']['train']['medical_vqa'],{'expected':2,'accepted':0})
            self.assertTrue(any('Missing' in item['reason'] for item in report['issues']))

    def test_stale_dependency_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder,rows=fixture(root)
            (folder/'all_annotations/labels.json').write_text('{}')
            _,report=self.build(root,{'train':rows,'validation':[],'test':[]})
            self.assertFalse(report['release_ready'])
            self.assertTrue(any('sha256' in item['reason'] for item in report['issues']))

    def test_mismatched_human_gt_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); _,rows=fixture(root)
            rows[0]['answer']='{}'
            _,report=self.build(root,{'train':rows,'validation':[],'test':[]})
            self.assertFalse(report['release_ready'])
            self.assertTrue(any('human' in item['reason'].lower() for item in report['issues']))

    def test_empty_rationale_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder,rows=fixture(root)
            path=folder/'GT/medical_spatiotemporal_vqa/severe_hemorrhage.json'
            target=json.loads(path.read_text()); target['response']['reasoning_process']=' '
            path.write_text(json.dumps(target))
            _,report=self.build(root,{'train':rows,'validation':[],'test':[]})
            self.assertFalse(report['release_ready'])

    def test_assembled_teacher_input_mismatch_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder,rows=fixture(root)
            path=folder/'GT/medical_spatiotemporal_vqa/severe_hemorrhage.json'
            target=json.loads(path.read_text())
            target['generation']['inputs']['assembled_input_sha256']='f'*64
            target['generation']['signature']=fingerprint(target['generation']['inputs'])
            path.write_text(json.dumps(target))
            _,report=self.build(root,{'train':rows,'validation':[],'test':[]})
            self.assertFalse(report['release_ready'])
            self.assertTrue(any('assembled_input' in item['reason'] for item in report['issues']))

    def test_pinned_manifest_tracks_human_gt_and_rejects_changed_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder,rows=fixture(root)
            _,report=self.build(root,{'train':rows,'validation':[],'test':[]})
            self.assertIn('dataset/06-native_media_dataset/abc/GT/medical_vqa.json',report['manifest']['files'])
            from scripts.train.shared.cot_data import verify_release_manifest
            path=root/'manifest.json'; path.write_text(json.dumps(report['manifest']))
            c=config(); c['cot'].update(release_manifest=str(path),release_manifest_sha256=sha256(path))
            verify_release_manifest(c,report)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA256'):
                verify_release_manifest(c,report)

    def test_manifest_identity_survives_dataset_relocation_but_not_target_changes(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'original'; root.mkdir()
            _,rows=fixture(root)
            _,original=self.build(root,{'train':rows,'validation':[],'test':[]})
            relocated=Path(directory)/'psc';shutil.copytree(root,relocated)
            moved=[dict(row,video=str(relocated/Path(row['video']).relative_to(root))) for row in rows]
            _,copied=self.build(relocated,{'train':moved,'validation':[],'test':[]})
            self.assertEqual(original['manifest'],copied['manifest'])
            from scripts.train.shared.cot_data import verify_release_manifest, pin_split_membership
            path=Path(directory)/'release.json';path.write_text(json.dumps(original['manifest']))
            c=config();c['cot'].update(release_manifest=str(path),release_manifest_sha256=sha256(path))
            verify_release_manifest(c,copied)
            target=relocated/'06-native_media_dataset/abc/GT/general_scene_description.json'
            value=json.loads(target.read_text());value['response']['answer']='A changed caption.';target.write_text(json.dumps(value))
            _,changed=self.build(relocated,{'train':moved,'validation':[],'test':[]})
            with self.assertRaisesRegex(ValueError,'pinned release'):
                verify_release_manifest(c,changed)
            for split in ('train','validation','test'):
                (root/(split+'.jsonl')).write_text('original membership')
                (relocated/(split+'.jsonl')).write_text('original membership')
            pin_split_membership(original,root);pin_split_membership(copied,relocated)
            self.assertEqual(original['manifest'],copied['manifest'])
            (relocated/'train.jsonl').write_text('changed membership')
            pin_split_membership(copied,relocated)
            self.assertNotEqual(original['manifest'],copied['manifest'])

    def test_teacherless_stabilized_heldout_media_are_pinned_and_changes_block_release(self):
        from types import SimpleNamespace
        from scripts.train.sft.launch import prepare_sft_data
        from scripts.train.tests.test_cot_integration import run_config
        from scripts.train.shared.cot_data import verify_release_manifest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); _,train=fixture(root)
            folder,held=fixture(root,variant='stabilized',clip_id='held',targets=False)
            for split,rows in [('train',train),('validation',held),('test',[])]:
                (root/(split+'.jsonl')).write_text('\n'.join(json.dumps(r['reference']) for r in rows))
            args=SimpleNamespace(media_root=root,split_dir=root,catalog=CATALOG,prompts=REPO/'scripts/train/prompts/answer_only/RGB/medical_vqa.yaml')
            c=compile_run(run_config())['experiment']
            splits,_,report=prepare_sft_data(args,c)
            self.assertTrue(report['release_ready'],report['issues'])
            self.assertEqual(len(splits['validation']),2)
            self.assertEqual(report['teacher_loss']['validation']['covered'],0)
            for name in ('held.mp4','held.json','provenance.json','all_annotations/meta.json','all_annotations/labels.json','all_annotations/camera_motion.json'):
                self.assertIn('dataset/07-stabilized_media_dataset/held/'+name,report['manifest']['files'])
            path=root/'release.json';path.write_text(json.dumps(report['manifest']))
            c['cot'].update(release_manifest=str(path),release_manifest_sha256=sha256(path))
            with (folder/'held.mp4').open('ab') as stream:
                stream.write(b'changed video bytes')
            changed_splits,_,changed=prepare_sft_data(args,c)
            self.assertEqual(len(changed_splits['validation']),2)
            self.assertFalse(changed['release_ready'])
            with self.assertRaisesRegex(ValueError,'pinned release'):
                verify_release_manifest(c,changed)

    def test_membership_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); _,rows=fixture(root)
            with self.assertRaisesRegex(ValueError,'split'):
                self.build(root,{'train':rows,'validation':rows,'test':[]})

if __name__=='__main__': unittest.main()
