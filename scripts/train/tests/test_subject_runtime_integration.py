import copy
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

class SubjectRuntimeTests(unittest.TestCase):
    def test_fingerprint_is_portable_but_tracks_target_and_content(self):
        from scripts.train.shared.runtime import data_fingerprint
        rows=[{'image':'/machine-a/crop.jpg','task':'subject_check','answer':'{"positive":0}', 'reference':{'image_sha256':'abc'}}]
        moved=copy.deepcopy(rows);moved[0]['image']='/machine-b/crop.jpg'
        self.assertEqual(data_fingerprint(rows),data_fingerprint(moved))
        moved[0]['reference']['image_sha256']='changed'
        self.assertNotEqual(data_fingerprint(rows),data_fingerprint(moved))
        moved=copy.deepcopy(rows);moved[0]['answer']='{"positive":1}'
        self.assertNotEqual(data_fingerprint(rows),data_fingerprint(moved))

    def test_saved_subject_rows_omit_local_paths_without_mutation(self):
        from scripts.train.shared.runtime import metadata_mixture_report
        report={'subject_check_evaluation':{'test':[{'image':'/local/private/crop.jpg','reference':{'image':'rgb/id/image.jpg','image_sha256':'abc'}}]}, 'subject_check':{'manifest_sha256':'def'}}
        saved=metadata_mixture_report(report)
        self.assertNotIn('image',saved['subject_check_evaluation']['test'][0])
        self.assertEqual(saved['subject_check_evaluation']['test'][0]['reference']['image_sha256'],'abc')
        self.assertIn('image',report['subject_check_evaluation']['test'][0])

    def test_image_processor_options_are_separate_from_video(self):
        from scripts.train.shared.modeling import build_processor
        fake=ModuleType('transformers')
        fake.AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(max_position_embeddings=8192)))
        fake.AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=object()))
        collator=ModuleType('scripts.train.shared.collator');collator.AnswerOnlyCollator=Mock()
        video=ModuleType('scripts.train.shared.video');video.spatial_options=Mock(return_value={'video_budget':123})
        config={'max_length':4096,'video':{'fps':5},'subject_check':{'enabled':True,'image_processor_kwargs':{'size': {'shortest_edge':1024, 'longest_edge':65536}}}}
        with patch.dict('sys.modules',{'transformers':fake,'scripts.train.shared.collator':collator,'scripts.train.shared.video':video}):
            build_processor({'local_path':'/unused'},config)
        self.assertEqual(collator.AnswerOnlyCollator.call_args.kwargs['image_processor_kwargs'],{'size': {'shortest_edge':1024, 'longest_edge':65536}})
        self.assertEqual(collator.AnswerOnlyCollator.call_args.args[3],{'video_budget':123})

class SubjectPreflightTests(unittest.TestCase):
    def test_missing_optional_dataset_fails_before_launch(self):
        import tempfile
        from pathlib import Path
        from scripts.train.run import preflight
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('train.jsonl','validation.jsonl','test.jsonl','catalog.yaml','prompts.yaml'):
                (root/name).write_text('')
            config={'subject_check':{'enabled':True,'modality':'rgb'}}
            paths={'media_root':str(root),'split_dir':str(root),'catalog':str(root/'catalog.yaml'),'prompts':str(root/'prompts.yaml')}
            plan={'run':{'paths':paths,'resume':None},'experiment':config,'stages':[]}
            with self.assertRaisesRegex(ValueError,'subject-check manifest'):
                preflight(plan)

if __name__=='__main__':unittest.main()
