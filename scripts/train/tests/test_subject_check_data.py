import json
from pathlib import Path
import unittest
import tempfile
from contextlib import contextmanager

@contextmanager
def raises(error, match=None):
    with unittest.TestCase().assertRaisesRegex(error, match or ".*"):
        yield
from scripts.train.shared import subject_check as sc
from scripts.train.shared.data import atomic_json, sha256


def fixture(tmp_path):
    dataset = tmp_path / '10-other_datasets/people_negatives_VK_20261102'
    entries=[]
    for split in ('train','validation','test'):
        folder=dataset/'rgb'/split
        folder.mkdir(parents=True)
        (folder/'image.jpg').write_bytes(b'fixture image')
        atomic_json(folder/'provenance.json', {'source_group':split,'image_sha256':sha256(folder/'image.jpg'),'source_record':{'checked':True,'check_by':'human_1','job_id':split,'modality':'rgb','split':'val' if split=='validation' else split,'cls':'non_person','group':split}})
        entries.append(dict(id=split,modality='rgb',split=split,source_group=split,image=f'rgb/{split}/image.jpg',provenance=f'rgb/{split}/provenance.json',image_sha256=sha256(folder/'image.jpg'),provenance_sha256=sha256(folder/'provenance.json')))
    atomic_json(dataset/'dataset_manifest.json',dict(schema_version=1,kind='subject_check_media',dataset=sc.DATASET,modality='rgb',entries=entries))
    split_dir=tmp_path/'08-train_val_test/date/answer_only'
    split_dir.mkdir(parents=True)
    for name in sc.SPLITS:
        (split_dir/(name+'.jsonl')).write_text('')
    atomic_json(split_dir/'summary.json',{'schema_version':1,'policy':{'grouping':'original_source_recording'},'pairs':{s:[] for s in sc.SPLITS},'split_sha256':{s:sha256(split_dir/(s+'.jsonl')) for s in sc.SPLITS}})
    return dataset,split_dir


def update_summary(split_dir, value):
    current=json.loads((split_dir/'summary.json').read_text())
    current.update(value)
    current['pairs']={name:current['pairs'].get(name,[]) for name in sc.SPLITS}
    current['split_sha256']={name:sha256(split_dir/(name+'.jsonl')) for name in sc.SPLITS}
    atomic_json(split_dir/'summary.json',current)


def test_mapping_requires_exactly_one_subject():
    for selected,expected in [('casualty',1),('medic',1),('robot',0),('other_object',0)]:
        answer={f'subject_is_{x}':int(x==selected) for x in ('casualty','medic','robot','other_object')}
        assert sc.derive_subject_answer(answer)[sc.SUBJECT_KEYS[0]]==expected
    with raises(ValueError):
        sc.derive_subject_answer(dict(subject_is_casualty=1,subject_is_robot=1))


def test_preparation_frozen_loader_and_tamper(tmp_path):
    dataset,split_dir=fixture(tmp_path)
    sc.prepare_negative_gt(tmp_path)
    sc.prepare_subject_check_splits(tmp_path,split_dir,modality='rgb')
    cfg={'enabled':True,'modality':'rgb','include_existing_subject_labels':False}
    rows,report=sc.load_subject_check_splits(tmp_path,split_dir,cfg,{},None,None)
    assert [len(rows[x]) for x in sc.SPLITS]==[1,1,1]
    assert json.loads(rows['train'][0]['answer'])==dict(subject_is_human_or_human_like_manikin=0,subject_is_not_human_or_manikin=1)
    assert rows['train'][0]['media_type']=='image'
    (dataset/'rgb/test/image.jpg').write_bytes(b'tampered')
    with raises(ValueError,match='hash|changed'):
        sc.load_subject_check_splits(tmp_path,split_dir,cfg,{},None,None)


def test_overlap_rejected_and_review_required(tmp_path):
    dataset,split_dir=fixture(tmp_path)
    update_summary(split_dir,{'pairs':{'train':[{'recording':'test'}]}})
    sc.prepare_negative_gt(tmp_path)
    with raises(ValueError,match='recording'):
        sc.prepare_subject_check_splits(tmp_path,split_dir,modality='rgb')
    path=dataset/'rgb/train/provenance.json'
    atomic_json(path,{'source_group':'train','source_record':{'checked':True,'check_by':'automatic'}})
    manifest=json.loads((dataset/'dataset_manifest.json').read_text())
    manifest['entries'][0]['provenance_sha256']=sha256(path)
    atomic_json(dataset/'dataset_manifest.json',manifest)
    with raises(ValueError,match='review'):
        sc.prepare_negative_gt(tmp_path)

class Tests(unittest.TestCase):
    def test_mapping(self):
        test_mapping_requires_exactly_one_subject()
    def test_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            test_preparation_frozen_loader_and_tamper(Path(d))
    def test_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            test_overlap_rejected_and_review_required(Path(d))

class MedicalTests(unittest.TestCase):
    def test_one_example_per_view_and_split_even_many_categories(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            dataset,split_dir=fixture(root)
            sc.prepare_negative_gt(root)
            medical={s:[] for s in sc.SPLITS}
            for variant,dirname in [('native','06-native_media_dataset'),('stabilized','07-stabilized_media_dataset')]:
                clip=root/dirname/'c1'
                atomic_json(clip/'all_annotations/meta.json',{'clip':'record_i1s1_C1','modality':'RGB'})
                atomic_json(clip/'all_annotations/labels.json',{})
                label_hash=sha256(clip/'all_annotations/labels.json')
                atomic_json(clip/'provenance.json',{'annotation_sha256':{'labels.json':label_hash}})
                atomic_json(clip/'GT/medical_vqa.json',{'source_hashes':{'labels_sha256':label_hash},'categories':{'subject_type':{'subject_is_casualty':1,'subject_is_medic':0,'subject_is_robot':0,'subject_is_other_object':0}}})
                sc.prepare_video_subject_gt(clip,{'subject_is_casualty':1,'subject_is_medic':0,'subject_is_robot':0,'subject_is_other_object':0},{'labels_sha256':label_hash})
                for category in ('subject_type','injury'):
                    medical['train'].append({'video':str(clip/'c1.mp4'),'reference':{'variant':variant,'clip':'c1','category':category}})
            sc.prepare_subject_check_splits(root,split_dir,modality='rgb')
            rows,report=sc.load_subject_check_splits(root,split_dir,{'enabled':True,'modality':'rgb'},medical,None,None)
            self.assertEqual(len(rows['train']),3)
            self.assertEqual(sum(r['media_type']=='video' for r in rows['train']),2)
            self.assertEqual(report['class_modality_counts']['train']['rgb']['positive'],2)

class IntegrityTests(unittest.TestCase):
    def test_incomplete_manifest_fails(self):
        with tempfile.TemporaryDirectory() as d:
            dataset,_=fixture(Path(d))
            manifest=json.loads((dataset/'dataset_manifest.json').read_text())
            manifest['complete']=False
            atomic_json(dataset/'dataset_manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'incomplete'):
                sc.prepare_negative_gt(Path(d))

    def test_overlap_uses_unselected_original_recording(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,split_dir=fixture(root)
            update_summary(split_dir,{'pairs':{'train':[{'recording':'test'}]}})
            sc.prepare_negative_gt(root)
            with self.assertRaisesRegex(ValueError,'recording'):
                sc.prepare_subject_check_splits(root,split_dir,modality='rgb')

class PromptTests(unittest.TestCase):
    def test_uses_approved_prompt_without_invented_viewpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,split_dir=fixture(root)
            sc.prepare_negative_gt(root)
            sc.prepare_subject_check_splits(root,split_dir,modality='rgb')
            rows,_=sc.load_subject_check_splits(root,split_dir,{'enabled':True,'modality':'rgb','include_existing_subject_labels':False},{},None,None)
            self.assertIn('green bounding box', rows['train'][0]['system'])
            self.assertIn('human-like manikin', rows['train'][0]['question'])
            self.assertNotIn('CAMERA VIEWPOINT', rows['train'][0]['question'])

class RecordingTests(unittest.TestCase):
    def test_prefixed_negative_recording_cannot_leak(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            dataset,split_dir=fixture(root)
            manifest=json.loads((dataset/'dataset_manifest.json').read_text())
            entry=manifest['entries'][2]
            entry['source_group']='night_ambush/P2D4_A8_S1'
            path=dataset/entry['provenance']
            provenance=json.loads(path.read_text())
            provenance['source_group']=provenance['source_record']['group']=entry['source_group']
            atomic_json(path,provenance)
            entry['provenance_sha256']=sha256(path)
            atomic_json(dataset/'dataset_manifest.json',manifest)
            update_summary(split_dir,{'pairs':{'train':[{'recording':'P2D4_A8_S1'}]}})
            sc.prepare_negative_gt(root)
            with self.assertRaisesRegex(ValueError,'recording'):
                sc.prepare_subject_check_splits(root,split_dir,modality='rgb')

class ModalityTests(unittest.TestCase):
    def test_missing_requested_modality_rejected_at_split_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            dataset,split_dir=fixture(root)
            manifest=json.loads((dataset/'dataset_manifest.json').read_text())
            manifest['modality']='rgb'
            atomic_json(dataset/'dataset_manifest.json',manifest)
            sc.prepare_negative_gt(root)
            sc.prepare_subject_check_splits(root,split_dir,modality='rgb')
            for modality in ('ir','both'):
                with self.subTest(modality=modality):
                    with self.assertRaisesRegex(ValueError,'modality'):
                        sc.prepare_subject_check_splits(root,split_dir,modality=modality,inspect=True)
                    with self.assertRaisesRegex(ValueError,'modality'):
                        sc.load_subject_check_splits(root,split_dir,{'enabled':True,'modality':modality,'include_existing_subject_labels':False},{},None,None)

class VideoGTTests(unittest.TestCase):
    def test_prepared_subject_target_pins_labels_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            clip=Path(d)
            atomic_json(clip/'all_annotations/labels.json',{})
            atomic_json(clip/'provenance.json',{})
            labels=sha256(clip/'all_annotations/labels.json')
            sc.prepare_video_subject_gt(clip,{'subject_is_casualty':1,'subject_is_medic':0,'subject_is_robot':0,'subject_is_other_object':0},{'labels_sha256':labels})
            target=json.loads((clip/'GT/subject_check.json').read_text())
            self.assertEqual(target['answer']['subject_is_human_or_human_like_manikin'],1)
            atomic_json(clip/'all_annotations/labels.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'labels'):
                sc.prepare_video_subject_gt(clip,{'subject_is_casualty':1,'subject_is_medic':0,'subject_is_robot':0,'subject_is_other_object':0},{'labels_sha256':labels})

class SourceIdentityTests(unittest.TestCase):
    def test_entry_cannot_reassign_reviewed_source_split(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            dataset,_=fixture(root)
            path=dataset/'dataset_manifest.json'
            manifest=json.loads(path.read_text())
            manifest['entries'][0]['split']='test'
            atomic_json(path,manifest)
            with self.assertRaisesRegex(ValueError,'identity'):
                sc.prepare_negative_gt(root)

class AttachTests(unittest.TestCase):
    def test_attach_preserves_medical_files_without_reselection(self):
        from scripts.train.splits.cli import main
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,split_dir=fixture(root)
            update_summary(split_dir,{'pairs':{s:[] for s in sc.SPLITS}})
            before={p.name:p.read_bytes() for p in split_dir.iterdir()}
            sc.prepare_negative_gt(root)
            with patch('scripts.train.splits.cli.load_inventory',side_effect=AssertionError('must not reselect')):
                self.assertEqual(main(['--media-root',str(root),'--attach-subject-check-to',str(split_dir),'--include-subject-check','--subject-check-modality','rgb']),0)
            self.assertEqual(before,{name:(split_dir/name).read_bytes() for name in before})
            self.assertTrue((split_dir/'subject_check/summary.json').is_file())

class SummaryTests(unittest.TestCase):
    def test_recording_membership_summary_is_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,split_dir=fixture(root)
            update_summary(split_dir,{'pairs':{s:[] for s in sc.SPLITS}})
            sc.prepare_negative_gt(root)
            sc.prepare_subject_check_splits(root,split_dir,modality='rgb')
            update_summary(split_dir,{'pairs':{'train':[{'recording':'new'}]}})
            with self.assertRaisesRegex(ValueError,'summary hash'):
                sc.load_subject_check_splits(root,split_dir,{'enabled':True,'modality':'rgb','include_existing_subject_labels':False},{},None,None)

class FrozenMedicalTests(unittest.TestCase):
    def test_subject_split_requires_original_recording_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,split_dir=fixture(root)
            (split_dir/'summary.json').unlink()
            sc.prepare_negative_gt(root)
            with self.assertRaisesRegex(ValueError,'frozen medical summary'):
                sc.prepare_subject_check_splits(root,split_dir,modality='rgb')

class SummaryShapeTests(unittest.TestCase):
    def test_real_summary_excluded_pairs_are_supported(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,split_dir=fixture(root)
            path=split_dir/'summary.json'
            summary=json.loads(path.read_text())
            summary['pairs']['excluded']=[]
            atomic_json(path,summary)
            sc.prepare_negative_gt(root)
            self.assertEqual(sc.prepare_subject_check_splits(root,split_dir,modality='rgb')['counts']['train'],1)

class RecordingAlignmentTests(unittest.TestCase):
    def test_aligns_crop_reference_without_changing_sources_or_medical_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset,split=fixture(root)
            update_summary(split,{'pairs':{'train':[{'recording':'test'}]}})
            sc.prepare_negative_gt(root)
            preserved=[dataset/'dataset_manifest.json',split/'summary.json',*[split/(s+'.jsonl') for s in sc.SPLITS]]
            hashes={p:sha256(p) for p in preserved}
            report=sc.prepare_subject_check_splits(root,split,modality='rgb',split_policy='match-medical')
            self.assertEqual(report['counts'],{'train':2,'validation':1,'test':0})
            self.assertEqual(report['reassigned_count'],1)
            rows,_=sc.load_subject_check_splits(root,split,{'enabled':True,'modality':'rgb','include_existing_subject_labels':False},{},None,None)
            moved=next(r['reference'] for r in rows['train'] if r['reference']['id']=='test')
            self.assertEqual(moved['original_split'],'test')
            self.assertEqual(moved['split'],'train')
            self.assertEqual(hashes,{p:sha256(p) for p in preserved})
    def test_loader_rejects_forged_original_split_even_with_rehashed_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset,split=fixture(root)
            update_summary(split,{'pairs':{'train':[{'recording':'test'}]}})
            sc.prepare_negative_gt(root)
            sc.prepare_subject_check_splits(root,split,modality='rgb',split_policy='match-medical')
            p=split/'subject_check/train.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()]
            rows[0]['original_split']='validation';p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            summary=split/'subject_check/summary.json';doc=json.loads(summary.read_text());doc['split_sha256']['train']=sha256(p);atomic_json(summary,doc)
            with self.assertRaises(ValueError):sc.load_subject_check_splits(root,split,{'enabled':True,'modality':'rgb','include_existing_subject_labels':False},{},None,None)

class ImageDirectoryRecordingTests(unittest.TestCase):
    def test_val_directories_retain_recording_identity(self):
        owners={'Y3_workshop_UAV_RGB_(bag_a)_(camera)_stamp':'test'}
        a=sc._assigned_entry({'source_group':'/review/bag_a_eo/images/val','split':'train'},owners,'match-medical')
        self.assertEqual(a['split'],'test')
        self.assertNotEqual(sc._subject_group('/review/bag_a_eo/images/val',owners),sc._subject_group('/review/bag_b_eo/images/val',owners))

    def test_channels_and_image_splits_share_one_recording(self):
        self.assertEqual(sc._subject_group('/review/bag_a_eo/images/val',{}),sc._subject_group('/review/bag_a_eo_onboard/images/train',{}))

    def test_ambiguous_medical_recording_is_rejected(self):
        owners={'Y3_RGB_(bag_a)_(camera1)_stamp':'train','Y3_RGB_(bag_a)_(camera2)_stamp':'test'}
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            sc._subject_group('/review/bag_a_eo/images/val',owners)
