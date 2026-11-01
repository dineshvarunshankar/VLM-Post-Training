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

    def test_reject_overlapping_gpu_pair_before_container(self):
        env=dict(os.environ,ABCI_TRAIN_GPU='2',ABCI_EVAL_GPU='2')
        result=subprocess.run(['bash',str(SCRIPT),'check','unused.yaml'],env=env,capture_output=True,text=True)
        self.assertIn('distinct',result.stderr)

    def test_reject_gpu_outside_node(self):
        env=dict(os.environ,ABCI_TRAIN_GPU='8')
        result=subprocess.run(['bash',str(SCRIPT),'check','unused.yaml'],env=env,capture_output=True,text=True)
        self.assertIn('0 to 7',result.stderr)


class ABCIMultiGPULauncherTests(unittest.TestCase):
    def check_devices(self, training, evaluation):
        env = dict(os.environ, ABCI_TRAIN_GPU=training, ABCI_EVAL_GPU=evaluation)
        return subprocess.run(['bash', str(SCRIPT), 'check', '/missing/run.yaml'],
                              env=env, capture_output=True, text=True)

    def test_four_distinct_training_gpus_reach_file_preflight(self):
        result = self.check_devices('0,1,2,3', '4')
        self.assertNotIn('GPU indices', result.stderr)
        self.assertIn('No such file', result.stderr)

    def test_reject_duplicate_training_devices(self):
        self.assertIn('duplicate', self.check_devices('0,1,1,3', '4').stderr)

    def test_reject_eval_device_in_training_group(self):
        self.assertIn('distinct', self.check_devices('0,1,2,3', '2').stderr)
