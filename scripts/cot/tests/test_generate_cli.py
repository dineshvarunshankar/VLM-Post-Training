"""Offline CLI coverage using prepared-manifest fixtures; no API calls."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import sys
import unittest
from unittest.mock import patch
import yaml

import test_preview
import generate


class GenerateCliTests(unittest.TestCase):
    def setUp(self):
        test_preview.PreviewTests.setUp(self)
        # This fixture contains one category, so use its matching catalog subset.
        catalog = yaml.safe_load(generate.CATALOG.read_text())
        catalog['categories'] = {'severe_hemorrhage': catalog['categories']['severe_hemorrhage']}
        self.catalog = self.root / 'catalog.yaml'
        self.catalog.write_text(yaml.safe_dump(catalog))

    def invoke(self, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ['generate.py', '--dataset-root', str(self.root), *options]
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {}, clear=True), \
                patch('generate.load_dotenv'), \
                patch('generate.CATALOG', self.catalog), patch('medical.CATALOG', self.catalog), \
                patch('medical.frame_times', return_value=[i / 10 for i in range(50)]), \
                patch('generate.generate', side_effect=AssertionError('API calls forbidden')), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = generate.main()
            except SystemExit as error:
                code = error.code
        rows = [json.loads(line) for line in stdout.getvalue().splitlines()]
        return code, rows, stderr.getvalue()

    def test_all_tasks_inspect_without_writes(self):
        code, rows, errors = self.invoke('--tasks', 'all', '--inspect')
        self.assertEqual(code, 0, errors)
        self.assertEqual([row['task'] for row in rows if 'system_prompt' in row], [
            'native_general_scene_description', 'native_medical_spatiotemporal_vqa',
            'stabilized_general_scene_description', 'stabilized_medical_spatiotemporal_vqa'])
        self.assertEqual(len([row for row in rows if row.get('status') == 'would_generate']), 4)
        self.assertFalse(list(self.root.glob('*/example/GT')))

    def test_selected_medical_tasks_use_category_filter(self):
        code, rows, errors = self.invoke('--tasks', 'stabilized_medical_spatiotemporal_vqa',
                                        'native_medical_spatiotemporal_vqa',
                                        '--category', 'severe_hemorrhage', '--inspect')
        self.assertEqual(code, 0, errors)
        requests = [row for row in rows if 'system_prompt' in row]
        self.assertEqual([row['task'] for row in requests], [
            'stabilized_medical_spatiotemporal_vqa', 'native_medical_spatiotemporal_vqa'])
        self.assertEqual([row['category'] for row in requests], ['severe_hemorrhage'] * 2)

    def test_same_variant_clip_and_duplicate_task(self):
        code, rows, errors = self.invoke('--tasks', 'native_general_scene_description',
                                        'native_medical_spatiotemporal_vqa',
                                        'native_general_scene_description', '--clip', 'example', '--inspect')
        self.assertEqual(code, 0, errors)
        self.assertEqual(len([row for row in rows if 'system_prompt' in row]), 2)

    def test_invalid_combinations_fail_before_processing(self):
        cases = [
            ('--tasks', 'all', '--clip', 'example'),
            ('--tasks', 'all', '--category', 'severe_hemorrhage'),
            ('--tasks', 'all', 'native_general_scene_description'),
            ('--task', 'native_general_scene_description', '--tasks', 'all'),
        ]
        for options in cases:
            with self.subTest(options=options):
                code, rows, _ = self.invoke(*options)
                self.assertEqual(code, 2)
                self.assertEqual(rows, [])

    def test_stops_before_next_task_after_failure(self):
        (self.root / '06-native_media_dataset/example/provenance.json').write_text('{}')
        code, rows, errors = self.invoke('--tasks', 'native_general_scene_description',
                                        'stabilized_general_scene_description')
        self.assertEqual(code, 1, errors)
        self.assertEqual(rows, [])
        self.assertIn('Prepared clip differs', errors)

    def test_existing_single_task_and_default_remain_supported(self):
        for options, expected in [((), 'native_general_scene_description'),
                                  (('--task', 'stabilized_general_scene_description'),
                                   'stabilized_general_scene_description')]:
            with self.subTest(options=options):
                code, rows, errors = self.invoke(*options, '--inspect')
                self.assertEqual(code, 0, errors)
                self.assertEqual([row['task'] for row in rows if 'system_prompt' in row], [expected])
