import os
from pathlib import Path
import subprocess
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'launch_abci.sh'

class ABCILauncherTests(unittest.TestCase):
    def run_script(self, *args):
        env = dict(os.environ)
        env.pop('PBS_JOBID', None)
        return subprocess.run(['bash', str(SCRIPT), *args], env=env, capture_output=True, text=True)

    def test_no_implicit_training(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn('check|server|train', result.stderr)

    def test_training_requires_scheduler_allocation(self):
        result = self.run_script('train', 'unused.yaml')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('PBS allocation', result.stderr)

    def test_server_requires_scheduler_allocation(self):
        result = self.run_script('server', 'unused.yaml')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('PBS allocation', result.stderr)
