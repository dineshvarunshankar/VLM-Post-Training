import json
import tempfile
import unittest
from pathlib import Path
from scripts.train.shared import data

class EvaluationReferencesTests(unittest.TestCase):
    def fixture(self, root):
        ao=root/'answer_only';cot=root/'cot';ao.mkdir();cot.mkdir()
        for name in ('train','validation','test'):
            (ao/(name+'.jsonl')).write_text('full\n')
            (cot/(name+'.jsonl')).write_text('filtered\n')
        (ao/'summary.json').write_text('{}')
        (cot/'summary.json').write_text(json.dumps({'answer_format':'cot','parent_split':'../answer_only','parent_split_sha256':{p.name:data.sha256(p) for p in ao.iterdir()}}))
        return ao,cot

    def test_keeps_filtered_training_and_full_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            ao,cot=self.fixture(Path(d))
            paths=data.split_reference_paths(cot)
            self.assertEqual(paths['train'],cot/'train.jsonl')
            self.assertEqual(paths['validation'],ao/'validation.jsonl')
            self.assertEqual(paths['test'],ao/'test.jsonl')

    def test_parent_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ao,cot=self.fixture(Path(d));(ao/'test.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'parent'):
                data.split_reference_paths(cot)

    def test_regular_split_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertEqual(data.split_reference_paths(root),{s:root/(s+'.jsonl') for s in ('train','validation','test')})

    def test_loader_and_builder_keep_full_teacherless_heldout(self):
        from copy import deepcopy
        from scripts.train.tests.test_cot_data import fixture, CATALOG, config
        from scripts.train.shared.cot_data import build_cot_splits, pin_split_membership, ASSESSMENT
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ao=root/'answer_only';cot=root/'cot';ao.mkdir();cot.mkdir()
            rows={}
            for split in ('train','validation','test'):
                _,rows[split]=fixture(root,variant='stabilized',clip_id=split,targets=split=='train')
                references=[r['reference'] for r in rows[split]]
                (ao/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in references))
                (cot/(split+'.jsonl')).write_text(json.dumps(references[0])+'\n')
            summary={'schema_version':1,'policy':{'grouping':'original_source_recording'},'modality':'RGB',
                     'catalog_sha256':data.sha256(CATALOG),
                     'pairs':{s:[{'native_id':'unused_'+s,'stabilized_id':s,'recording':s}] for s in rows},
                     'split_sha256':{s:data.sha256(ao/(s+'.jsonl')) for s in rows}}
            (ao/'summary.json').write_text(json.dumps(summary))
            child=deepcopy(summary)
            child.update(answer_format='cot',parent_split='../answer_only',
                         parent_split_sha256={p.name:data.sha256(p) for p in ao.iterdir()},
                         split_sha256={s:data.sha256(cot/(s+'.jsonl')) for s in rows})
            (cot/'summary.json').write_text(json.dumps(child))
            loaded=data.load_splits(root,cot,CATALOG,ASSESSMENT,sources={'native':False,'stabilized':True})
            self.assertEqual({s:len(v) for s,v in loaded.items()},{'train':1,'validation':2,'test':2})
            built,report=build_cot_splits(loaded,media_root=root,catalog=data.read_yaml(CATALOG),config=config())
            self.assertEqual(len(built['test']),2)
            self.assertEqual([r['answer'] for r in built['test']],[r['answer'] for r in rows['test']])
            self.assertTrue(all(not r['teacher_available'] for r in built['test']))
            self.assertTrue(report['release_ready'])
            pin_split_membership(report,cot)
            self.assertEqual(report['manifest']['files']['splits/parent/test.jsonl'],data.sha256(ao/'test.jsonl'))
            self.assertEqual(report['source_paths']['files']['splits/parent/summary.json'],str((ao/'summary.json').resolve()))

    def test_parent_path_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ao,cot=self.fixture(Path(d))
            path=cot/'summary.json';summary=json.loads(path.read_text());summary['parent_split']='../../another_release/answer_only';path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'parent'):
                data.split_reference_paths(cot)
            summary['parent_split']='../answer_only';path.write_text(json.dumps(summary))
            original=ao/'test.jsonl';target=ao/'saved-test.jsonl';original.rename(target);original.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'parent'):
                data.split_reference_paths(cot)
