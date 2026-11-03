"""Metadata-only bounds must follow the production sampler and selected sources."""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from scripts.train.shared.data import DATASETS

REPO = Path(__file__).resolve().parents[3]


class FrameBoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / 'media'
        self.splits = self.root / 'splits'
        self.splits.mkdir()
        self.video = {'fps': 5, 'gemma': {'max_soft_tokens': 140}, 'qwen': {'max_pixels': 251658240},
                      'sources': {source: {'fps': fps, 'max_soft_tokens': 140, 'max_pixels_per_frame': 322560}
                                  for source, fps in [('native', 2), ('stabilized', 5)]}}
        self.plan = {'run': {'paths': {'media_root': str(self.media), 'split_dir': str(self.splits)},
                             'dataset': {'sources': ['native', 'stabilized']}},
                     'experiment': {'video': self.video}}
        self.metadata('native', 'slow', 199, 1.5)
        self.metadata('stabilized', 'fractional', 1000, 29.97)
        self.metadata('stabilized', 'short', 1, 30)
        for split, rows in {'train': [('native', 'slow'), ('stabilized', 'short')],
                            'validation': [('stabilized', 'fractional')],
                            'test': [('native', 'slow')]}.items():
            (self.splits / f'{split}.jsonl').write_text(''.join(json.dumps({'variant': v, 'clip': c})+'\n' for v, c in rows))

    def metadata(self, variant, clip, frames, fps):
        path = self.media / DATASETS[variant] / clip / 'all_annotations/meta.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'n_frames': frames, 'fps': fps}))
        return path

    def measure(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.train.shared.frame_bound'),
                             'The shared metadata-only bound helper must exist')
        return importlib.import_module('scripts.train.shared.frame_bound').measure_frame_bound(self.plan)

    def test_union_uses_source_fps_and_keeps_bound_above_150_without_decoding(self):
        before = copy.deepcopy(self.plan)
        report = self.measure()
        self.assertEqual(report['profile_video_frames'], 199)
        self.assertEqual(report['clips'], 3)
        self.assertEqual(report['source_maxima'], {'native': 199, 'stabilized': 166})
        counts = {row['clip']: row['sampled_frames'] for row in report['maximum_cases']}
        self.assertEqual(counts, {'slow': 199, 'fractional': 166, 'short': 1})
        self.assertEqual(self.plan, before)
        self.assertEqual(report['split_sha256']['validation'], hashlib.sha256((self.splits/'validation.jsonl').read_bytes()).hexdigest())
        self.assertEqual(len(report['metadata_sha256']), 64)
        self.assertEqual(len(report['sampling_sha256']), 64)
        # There are deliberately no video files in this fixture.
        self.assertEqual(list(self.media.rglob('*.mp4')), [])

    def test_filtered_source_metadata_is_not_required(self):
        self.plan['run']['dataset']['sources'] = ['stabilized']
        (self.media / DATASETS['native'] / 'slow/all_annotations/meta.json').unlink()
        report = self.measure()
        self.assertEqual(report['profile_video_frames'], 166)
        self.assertEqual(report['source_maxima'], {'stabilized': 166})
        self.assertEqual(report['clips'], 2)

    def test_eval_only_native_clip_sets_bound_without_including_native_training(self):
        self.plan['run']['dataset']['sources'] = ['stabilized']
        self.plan['experiment']['evaluation'] = {'sources': ['native', 'stabilized']}
        self.metadata('native', 'validation_only', 251, 1.5)
        # This unselected native training clip must not be read, even though native evaluates.
        (self.media / DATASETS['native'] / 'slow/all_annotations/meta.json').unlink()
        (self.splits / 'validation.jsonl').write_text(
            json.dumps({'variant': 'native', 'clip': 'validation_only'})+'\n'+
            json.dumps({'variant': 'stabilized', 'clip': 'fractional'})+'\n')
        (self.splits / 'test.jsonl').write_text('')
        report = self.measure()
        self.assertEqual(report['profile_video_frames'], 251)
        self.assertEqual(report['source_maxima'], {'stabilized': 166, 'native': 251})
        self.assertEqual(report['clips'], 3)
        self.assertEqual(report['split_sources'], {'train': ['stabilized'],
                          'validation': ['native', 'stabilized'], 'test': ['native', 'stabilized']})

    def test_sampling_fingerprint_tracks_split_source_policy_without_changing_union(self):
        self.plan['run']['dataset']['sources'] = ['stabilized', 'native']
        first = self.measure()
        self.plan['run']['dataset']['sources'] = ['stabilized']
        self.plan['experiment']['evaluation'] = {'sources': ['native', 'stabilized']}
        second = self.measure()
        self.assertEqual(first['source_maxima'], second['source_maxima'])
        self.assertEqual(first['metadata_sha256'], second['metadata_sha256'])
        self.assertNotEqual(first['sampling_sha256'], second['sampling_sha256'])

    def test_nonmax_metadata_and_sampling_changes_are_fingerprinted(self):
        first = self.measure()
        self.metadata('stabilized', 'short', 2, 30)
        second = self.measure()
        self.assertEqual(first['profile_video_frames'], second['profile_video_frames'])
        self.assertNotEqual(first['metadata_sha256'], second['metadata_sha256'])
        self.video['sources']['stabilized']['fps'] = 4
        self.assertNotEqual(second['sampling_sha256'], self.measure()['sampling_sha256'])

    def test_minimum_profile_is_two_even_when_sampler_returns_one(self):
        self.plan['run']['dataset']['sources'] = ['stabilized']
        self.metadata('stabilized', 'fractional', 1, 29.97)
        self.assertEqual(self.measure()['profile_video_frames'], 2)

    def test_missing_or_invalid_metadata_fails_with_clip_path(self):
        path = self.media / DATASETS['native'] / 'slow/all_annotations/meta.json'
        for metadata in [None, {}, {'n_frames': 0, 'fps': 5}, {'n_frames': True, 'fps': 5},
                         {'n_frames': 3.5, 'fps': 5}, {'n_frames': 2, 'fps': 0},
                         {'n_frames': 2, 'fps': float('nan')}, {'n_frames': 2, 'fps': '5'}]:
            with self.subTest(metadata=metadata):
                if metadata is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'slow.*meta.json'):
                    self.measure()

    def test_missing_split_or_empty_selected_source_fails_clearly(self):
        (self.splits / 'test.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'test.jsonl'):
            self.measure()
        (self.splits / 'test.jsonl').write_text('')
        (self.splits / 'train.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, 'native'):
            self.measure()

    def write_config(self, name):
        original = REPO / 'scripts/train/configs' / name
        config = yaml.safe_load(original.read_text())
        # Resolve other config paths before relocating, but exercise relative data paths.
        for key, value in config['paths'].items():
            config['paths'][key] = str((original.parent / (value or ('fixture-'+key))).resolve())
        config['paths'].update(media_root='media', split_dir='splits')
        config['dataset']['sources'] = ['native', 'stabilized']
        config['shared']['video'] = self.video
        path = self.root / 'run.yaml'
        path.write_text(yaml.safe_dump(config))
        return path

    def test_shared_and_psc_clis_match_for_relative_paths_and_cot(self):
        for name in ['runs/gemma12b_sft_native_stabilized_medical_camera_viewpoint.yaml', 'cot/gemma4-12b-it.yaml']:
            with self.subTest(config=name):
                config = self.write_config(name)
                reports = []
                for module in ['scripts.train.shared.frame_bound', 'scripts.train.psc.frame_bound']:
                    output = self.root / (module+'.json')
                    result = subprocess.run([sys.executable, '-m', module, '--config', str(config), '--output', str(output)],
                                            cwd=REPO, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    reports.append(json.loads(result.stdout))
                    self.assertEqual(reports[-1], json.loads(output.read_text()))
                self.assertEqual(reports[0], reports[1])
                self.assertEqual(reports[0]['profile_video_frames'], 199)

    def test_psc_import_does_not_parse_process_arguments_or_touch_data(self):
        result = subprocess.run([sys.executable, '-c', 'import scripts.train.psc.frame_bound', '--unrelated'],
                                cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')


class ABCIFrameBoundModeTests(unittest.TestCase):
    setUp = FrameBoundTests.setUp
    metadata = FrameBoundTests.metadata
    write_config = FrameBoundTests.write_config
    # Reuse the tiny inventory fixture, without inheriting its test cases.
    def test_container_mode_runs_only_measurement_without_gpu_or_preflight(self):
        import shlex
        config = self.write_config('cot/gemma4-12b-it.yaml')
        work = self.root / 'work'
        repo = work / 'triage/repo'
        launcher = repo / 'scripts/train/launch_abci.sh'
        launcher.parent.mkdir(parents=True)
        launcher.write_text((REPO/'scripts/train/launch_abci.sh').read_text())
        python = work / 'env/vlm-post-training/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('#!/bin/bash\n' +
            f'printf "%s\\n" "$*" >> {shlex.quote(str(self.root/"python-calls"))}\n' +
            f'exec env PYTHONPATH={shlex.quote(str(REPO))} {shlex.quote(sys.executable)} "$@"\n')
        python.chmod(0o755)
        singularity = self.root / 'singularity'
        singularity.write_text('#!/bin/bash\n' +
            f'printf "%s\\n" "$*" > {shlex.quote(str(self.root/"container-args"))}\n' +
            'while [[ "$1" != bash ]]; do shift; done\nexec "$@"\n')
        singularity.chmod(0o755)
        image = self.root / 'fixture.sif'
        image.touch()
        env = dict(os.environ, ABCI_IMAGE=str(image), PATH=str(self.root)+os.pathsep+os.environ['PATH'])
        env.pop('PBS_JOBID', None)
        result = subprocess.run(['bash', str(launcher), 'frame-bound', str(config)], cwd=self.root,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['profile_video_frames'], 199)
        self.assertNotIn('--nv', (self.root/'container-args').read_text())
        calls = (self.root/'python-calls').read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].startswith('-m scripts.train.shared.frame_bound --config '), calls)
