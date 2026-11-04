import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.train import transfer_dataset as transfer
from scripts.train.storage import dataset_transfer as cloud


class SubjectTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.split = self.root / '08-train_val_test/release'
        self.split.mkdir(parents=True)
        prepared = {}
        for variant, folder in transfer.DATASETS.items():
            path = self.root / folder / 'dataset_manifest.json'
            path.parent.mkdir()
            path.write_text(json.dumps(dict(schema_version=1, variant=variant, clips=[])))
            prepared[variant] = {'sha256': transfer.digest(path)}
        hashes = {}
        for name in ('train', 'validation', 'test'):
            path = self.split / (name + '.jsonl')
            path.write_text('')
            hashes[name] = transfer.digest(path)
        (self.split / 'summary.json').write_text(json.dumps(dict(schema_version=1,policy={'grouping':'original_source_recording'},pairs={name:[] for name in ('train','validation','test')},prepared_manifests=prepared, split_sha256=hashes)))

    def test_disabled_transfer_excludes_optional_sidecars(self):
        sidecar = self.split / 'subject_check'
        sidecar.mkdir()
        (sidecar / 'summary.json').write_text('not prepared yet')
        plan = transfer.prepare_transfer(self.root, self.split)
        self.assertFalse(any('subject_check' in name for name in plan.files))

    def test_enabled_transfer_requires_prepared_subject_dataset(self):
        with self.assertRaisesRegex(ValueError, '[Ss]ubject|[Mm]issing'):
            transfer.prepare_transfer(self.root, self.split, include_subject_check=True)

    def test_cloud_subject_paths_require_opt_in_and_are_narrow(self):
        record = {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
        manifest = {'schema_version': 2, 'files': {'10-other_datasets/people_negatives_VK_20261102/rgb/box/image.jpg': record}}
        with self.assertRaises(ValueError):
            cloud.validate_manifest(manifest)
        self.assertEqual(cloud.validate_manifest(manifest, include_subject_check=True), manifest)
        manifest['files'] = {'10-other_datasets/unrelated/private.json': record}
        with self.assertRaises(ValueError):
            cloud.validate_manifest(manifest, include_subject_check=True)

    def subject_fixture(self):
        from scripts.train.shared.subject_check import prepare_negative_gt, prepare_subject_check_splits
        dataset = self.root / '10-other_datasets/people_negatives_VK_20261102'
        image = dataset / 'rgb/box/image.jpg'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'image')
        provenance = image.parent / 'provenance.json'
        provenance.write_text(json.dumps({'source_group':'recording','image_sha256':transfer.digest(image),'source_record':{'checked':True,'check_by':'human reviewer','job_id':'box','modality':'rgb','split':'train','cls':'other_not_person','group':'recording'}}))
        entry = dict(id='box',modality='rgb',split='train',source_group='recording',image='rgb/box/image.jpg',provenance='rgb/box/provenance.json',image_sha256=transfer.digest(image),provenance_sha256=transfer.digest(provenance))
        (dataset / 'dataset_manifest.json').write_text(json.dumps(dict(schema_version=1,kind='subject_check_media',modality='rgb',dataset='people_negatives_VK_20261102',entries=[entry])))
        prepare_negative_gt(self.root)
        prepare_subject_check_splits(self.root,self.split,modality='rgb')
        return image

    def test_opt_in_transfers_frozen_media_and_verifies_all_targets(self):
        image = self.subject_fixture()
        plan = transfer.prepare_transfer(self.root,self.split,include_subject_check=True)
        self.assertIn(str(image.relative_to(self.root)), plan.files)
        self.assertIn('08-train_val_test/release/subject_check/train.jsonl',plan.files)
        self.assertIn(str(image.relative_to(self.root)),plan.checksums)
        (image.parent / 'GT/subject_check.json').write_text('{}')
        with self.assertRaises(ValueError):
            transfer.prepare_transfer(self.root,self.split,include_subject_check=True)

    def test_opt_in_rejects_missing_frozen_membership_even_with_rehashed_sidecars(self):
        self.subject_fixture()
        train = self.split / 'subject_check/train.jsonl'
        train.write_text('')
        summary = self.split / 'subject_check/summary.json'
        data = json.loads(summary.read_text())
        data['split_sha256']['train'] = transfer.digest(train)
        summary.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'missing'):
            transfer.prepare_transfer(self.root,self.split,include_subject_check=True)

    def test_sidecar_cloud_manifest_also_requires_opt_in(self):
        record = {'bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}
        manifest = {'schema_version':2,'files':{'08-train_val_test/release/subject_check/train.jsonl':record}}
        with self.assertRaises(ValueError):
            cloud.validate_manifest(manifest)
        cloud.validate_manifest(manifest, include_subject_check=True)

    def test_all_transferred_modalities_require_current_gt(self):
        image = self.subject_fixture()
        dataset = image.parents[2]
        manifest = dataset / 'dataset_manifest.json'
        data = json.loads(manifest.read_text())
        ir = dataset / 'ir/box'
        ir.mkdir(parents=True)
        (ir / 'image.jpg').write_bytes(image.read_bytes())
        provenance = json.loads((image.parent / 'provenance.json').read_text())
        provenance['source_record']['modality'] = 'ir'
        (ir / 'provenance.json').write_text(json.dumps(provenance))
        data['modality'] = 'both'
        data['entries'].append({**data['entries'][0], 'modality':'ir','image':'ir/box/image.jpg','provenance':'ir/box/provenance.json','provenance_sha256':transfer.digest(ir / 'provenance.json')})
        manifest.write_text(json.dumps(data))
        from scripts.train.shared.subject_check import prepare_negative_gt, prepare_subject_check_splits
        prepare_negative_gt(self.root)
        prepare_subject_check_splits(self.root,self.split,modality='rgb',refresh=True)
        (ir / 'GT/subject_check.json').write_text('{}')
        with self.assertRaises(ValueError):
            transfer.prepare_transfer(self.root,self.split,include_subject_check=True)

    def test_enabled_transfer_includes_validated_medical_subject_gt(self):
        clip = self.root / '06-native_media_dataset/abc'
        (clip / 'all_annotations').mkdir(parents=True)
        (clip / 'GT').mkdir()
        (clip / 'abc.mp4').write_bytes(b'video')
        (clip / 'abc.json').write_text('{}')
        (clip / 'all_annotations/meta.json').write_text(json.dumps({'modality':'rgb','clip':'medical-recording_i0s0_C0'}))
        labels = clip / 'all_annotations/labels.json'
        labels.write_text('{}')
        label_hash = transfer.digest(labels)
        (clip / 'provenance.json').write_text(json.dumps({'annotation_sha256':{'labels.json':label_hash}}))
        source_hashes = {'labels_sha256':label_hash}
        (clip / 'GT/medical_vqa.json').write_text(json.dumps({'source_hashes':source_hashes,'categories':{'subject_type':{'subject_is_casualty':1,'subject_is_medic':0,'subject_is_robot':0,'subject_is_other_object':0}}}))
        target = clip / 'GT/subject_check.json'
        target.write_text(json.dumps({'answer':{'subject_is_human_or_human_like_manikin':1,'subject_is_not_human_or_manikin':0},'source_hashes':{**source_hashes,'provenance_sha256':transfer.digest(clip / 'provenance.json')}}))
        manifest = clip.parent / 'dataset_manifest.json'
        manifest.write_text(json.dumps({'schema_version':1,'variant':'native','clips':[{'clip_id':'abc','clip':'recording','level':'minimal','provenance_sha256':transfer.digest(clip / 'provenance.json')}]}))
        train = self.split / 'train.jsonl'
        train.write_text(json.dumps({'variant':'native','clip':'abc','recording':'medical-recording'})+'\n')
        summary = self.split / 'summary.json'
        data = json.loads(summary.read_text())
        data['prepared_manifests']['native']['sha256'] = transfer.digest(manifest)
        data['split_sha256']['train'] = transfer.digest(train)
        data['pairs']['train'] = [{'native_id':'abc','stabilized_id':'abc','recording':'medical-recording'}]
        summary.write_text(json.dumps(data))
        self.subject_fixture()
        relative = str(target.relative_to(self.root))
        self.assertNotIn(relative,transfer.prepare_transfer(self.root,self.split).files)
        plan = transfer.prepare_transfer(self.root,self.split,include_subject_check=True)
        self.assertIn(relative,plan.files)
        self.assertIn(relative,plan.checksums)
        target.write_text('{}')
        with self.assertRaises(ValueError):
            transfer.prepare_transfer(self.root,self.split,include_subject_check=True)
        from scripts.train.storage.subject_transfer import subject_transfer_files
        self.assertNotIn(target, subject_transfer_files(self.root,self.split,include_existing_subject_labels=False))
        from scripts.train.run import preflight
        catalog = self.root / 'catalog.yaml'
        prompts = self.root / 'prompts.yaml'
        catalog.write_text('')
        prompts.write_text('')
        plan = {'run':{'paths':{'media_root':str(self.root),'split_dir':str(self.split),'catalog':str(catalog),'prompts':str(prompts)},'resume':None},'experiment':{'subject_check':{'enabled':True,'modality':'rgb','include_existing_subject_labels':False}},'stages':[]}
        preflight(plan)
