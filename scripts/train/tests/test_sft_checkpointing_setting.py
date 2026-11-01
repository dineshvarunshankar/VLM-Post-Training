from pathlib import Path
import unittest
from scripts.train.sft.launch import build_training_config

class CheckpointingSettingTests(unittest.TestCase):
    def test_explicit_off_and_legacy_default(self):
        for values, expected in (({},True),({'gradient_checkpointing':False},False),({'gradient_checkpointing':True},True)):
            with self.subTest(values=values):
                config=build_training_config(Path('/tmp/checkpointing-setting-test'),dict(use_cpu=True,**values))
                self.assertEqual(config.gradient_checkpointing,expected)
