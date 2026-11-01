import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.train.shared.tracking import configure_tracking

class TrackingAttemptTests(unittest.TestCase):
    def test_fresh_attempts_keep_name_but_use_distinct_ids(self):
        args=SimpleNamespace(model='qwen',run_date='2026-11-01',run_name='medical',resume=None)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'WANDB_MODE':'offline'},clear=True):
            os.environ['TORCHELASTIC_RUN_ID']='attempt-a'
            first=configure_tracking(args,'sft_only',directory)
            first_id=os.environ['WANDB_RUN_ID']
            configure_tracking(args,'sft_only',directory)
            self.assertEqual(first_id,os.environ['WANDB_RUN_ID'])
            os.environ['TORCHELASTIC_RUN_ID']='attempt-b'
            second=configure_tracking(args,'sft_only',directory)
            self.assertEqual(first['run_name'],second['run_name'])
            self.assertNotEqual(first_id,os.environ['WANDB_RUN_ID'])
            self.assertEqual(os.environ['WANDB_RESUME'],'never')

    def test_checkpoint_resume_restores_original_tracking_id(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'WANDB_MODE':'offline'},clear=True):
            root=Path(directory)
            (root/'tracking').mkdir()
            (root/'tracking'/'run_id.json').write_text(json.dumps({'id':'original-attempt'}))
            args=SimpleNamespace(model='qwen',run_date='2026-11-01',run_name='medical',resume=root/'checkpoints'/'checkpoint-10')
            configure_tracking(args,'sft_only',root)
            self.assertEqual(os.environ['WANDB_RUN_ID'],'original-attempt')
            self.assertEqual(os.environ['WANDB_RESUME'],'allow')

    def test_legacy_resume_preserves_deterministic_id(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'WANDB_MODE':'offline'},clear=True):
            args=SimpleNamespace(model='qwen',run_date='2026-11-01',run_name='medical',resume=Path(directory)/'checkpoint')
            result=configure_tracking(args,'sft_only',directory)
            self.assertEqual(os.environ['WANDB_RUN_ID'],hashlib.sha256(result['run_name'].encode()).hexdigest()[:24])
