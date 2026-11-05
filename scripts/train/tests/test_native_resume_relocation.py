"""Native resumes allow only verified byte-identical CoT checkout relocation."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import unittest
import tempfile
from unittest.mock import patch

from scripts.train.shared.cot_data import fingerprint
from scripts.train.shared.runtime import save_run_metadata


def native_resume(tmp_path, answer_format="cot"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ('catalog.yaml', 'prompts.yaml', 'train.jsonl'):
        (tmp_path / name).write_text('')
    old, new = [tmp_path / name for name in ('old-checkout', 'new-checkout')]
    files, mappings = {}, [{}, {}]
    for index in range(20):
        relative = f'scripts/cot/source{index}.py'
        contents = f'# audited source {index}\n'.encode()
        key = 'repository/' + relative
        files[key] = hashlib.sha256(contents).hexdigest()
        for root, mapping in zip((old, new), mappings):
            source = root / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(contents)
            mapping[key] = str(source)
    files['dataset/video.mp4'] = hashlib.sha256(b'video').hexdigest()
    for mapping in mappings:
        mapping['dataset/video.mp4'] = str(tmp_path / 'video.mp4')
    report = {'manifest': {'schema_version': 1, 'files': files, 'sha256': fingerprint(files)},
              'source_paths': {'repository_root': str(old), 'dataset_root': str(tmp_path), 'files': mappings[0]}}
    args = SimpleNamespace(model='gemma4-31b-it', run_location={'recipe':'sft_only','answer_format':answer_format},
                           resume=None, catalog=tmp_path/'catalog.yaml', prompts=tmp_path/'prompts.yaml', split_dir=tmp_path)
    train = {'num_train_epochs': 8, 'learning_rate': 2e-5, 'lr_scheduler_type':'cosine'}
    config = {'answer_format':answer_format, 'training':train, 'evaluation':{'max_new_tokens':10},
              'cot':{'thinking_token_budget':0,'medical_max_new_tokens':1032,'caption_max_new_tokens':16}}
    model = {'generation':dict(do_sample=False,temperature=1.,top_p=1.,top_k=0,min_p=0.,presence_penalty=0.,repetition_penalty=1.)}
    output = tmp_path/'output'
    save_run_metadata(args,output,model,config,train,{}, {'train':[]},report)
    original = {p.name:p.read_bytes() for p in output.iterdir() if p.is_file()}
    checkpoint = output/'checkpoints/checkpoint-2'
    checkpoint.mkdir(parents=True)
    args.resume = checkpoint
    relocated = copy.deepcopy(report)
    relocated['source_paths'].update(repository_root=str(new),files=mappings[1])
    def save(value=relocated, *, current_config=None, current_train=None, splits=None):
        return save_run_metadata(args,output,model,current_config or config,current_train or train,{},
                                 {'train':[]} if splits is None else splits,value)
    return SimpleNamespace(save=save, report=relocated, original_report=report, output=output,
                           original=original, config=config, train=train, old=old,new=new)


def assert_original(fixture):
    for name, contents in fixture.original.items():
        assert (fixture.output/name).read_bytes() == contents


class NativeResumeRelocationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.environment = patch.dict('os.environ', {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        import os
        os.environ.pop('TRIAGE_RUN_CONFIG', None)
        self.fixture = native_resume(Path(self.directory.name))

    def test_verified_twenty_files_relocate_without_writes(self):
        with patch('scripts.train.shared.runtime.atomic_json', side_effect=AssertionError('native resume must not rewrite')):
            self.assertIsNone(self.fixture.save())
        assert_original(self.fixture)

    def test_exact_resume_does_not_rewrite(self):
        with patch('scripts.train.shared.runtime.atomic_json', side_effect=AssertionError('redundant rewrite')):
            self.assertIsNone(self.fixture.save(self.fixture.original_report))
        assert_original(self.fixture)

    def test_exact_answer_only_resume_does_not_rewrite(self):
        fixture = native_resume(Path(self.directory.name)/'answer-only', answer_format='answer_only')
        with patch('scripts.train.shared.runtime.atomic_json', side_effect=AssertionError('redundant rewrite')):
            self.assertIsNone(fixture.save(fixture.original_report))
        assert_original(fixture)

    def reject(self, change):
        fixture = self.fixture
        report = copy.deepcopy(fixture.report)
        config, train = copy.deepcopy(fixture.config), copy.deepcopy(fixture.train)
        key = 'repository/scripts/cot/source0.py'
        if change == 'file': Path(report['source_paths']['files'][key]).write_text('changed')
        elif change == 'symlink':
            source = Path(report['source_paths']['files'][key]); source.unlink(); source.symlink_to(fixture.old/'scripts/cot/source0.py')
        elif change == 'hash': report['manifest']['files'][key] = '0'*64
        elif change == 'data_path': report['source_paths']['files']['dataset/video.mp4'] = '/different/video.mp4'
        elif change == 'path': report['source_paths']['files'][key] = str(fixture.new/'scripts/cot/source1.py')
        elif change == 'epochs': train['num_train_epochs'] = 9; config['training'] = train
        elif change == 'learning_rate': train['learning_rate'] = 1e-5; config['training'] = train
        elif change in ('library','endpoint'):
            previous = json.loads((fixture.output/'run.json').read_text())
            if change == 'library': previous['libraries']['trl'] = 'different'
            else: previous['launch_config'] = {'execution':{'vllm_server_url':'http://original'}}
            (fixture.output/'run.json').write_text(json.dumps(previous))
            fixture.original['run.json'] = (fixture.output/'run.json').read_bytes()
        with self.assertRaises(ValueError):
            fixture.save(report,current_config=config,current_train=train,
                         splits={'train':[{'label':'changed'}]} if change == 'dataset' else None)
        assert_original(fixture)


for _change in ('file','symlink','hash','data_path','path','dataset','epochs','learning_rate','library','endpoint'):
    setattr(NativeResumeRelocationTests, 'test_reject_' + _change,
            lambda self, change=_change: self.reject(change))

if __name__ == '__main__':
    unittest.main()
