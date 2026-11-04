"""PBS startup must preserve the approved shared-evaluation allocation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.train.psc import launch


class ABCIMultinodeTests(unittest.TestCase):
    def test_pbs_accepts_committed_cot_config_before_mpi_handoff(self):
        script = Path(__file__).resolve().parents[1] / 'abci_sft_multinode.pbs'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root/'triage'/'repo'
            config = repo/'scripts/train/configs/cot/qwen3.5-9b.yaml'
            config.parent.mkdir(parents=True)
            config.write_text('schema_version: 2\n')
            subprocess.run(['git','init','-q',str(repo)],check=True)
            subprocess.run(['git','add','.'],cwd=repo,check=True)
            subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=repo,check=True)
            commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
            bins = root/'bin'; bins.mkdir()
            for name, content in [('module', '#!/bin/sh\nexit 0\n'), ('mpirun', '#!/bin/sh\nprintf "%s\\n" "$@" > "$MPI_CAPTURE"\n')]:
                path=bins/name; path.write_text(content); path.chmod(0o700)
            nodefile=root/'nodes'; nodefile.write_text('node1\n')
            capture=root/'mpi-args'
            env={**os.environ,'PATH':str(bins)+os.pathsep+os.environ['PATH'],
                 'PBS_JOBID':'1.pbs1','PBS_NODEFILE':str(nodefile),'PBS_O_WORKDIR':str(repo),
                 'RTYPE':'rt_HF','ABCI_SFT_CONFIG':str(config),'ABCI_SFT_COMMIT':commit,
                 'ABCI_TRAIN_NODES':'1','MPI_CAPTURE':str(capture)}
            result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(str(config),capture.read_text())

    def test_pbs_rank_mapping_and_missing_rank_rejection(self):
        resolve = getattr(launch, 'scheduler_identity', None)
        self.assertIsNotNone(resolve, 'PBS scheduler mapping is missing')
        self.assertEqual(resolve('pbs', {'PBS_JOBID':'2843000.pbs1', 'TRIAGE_NODE_RANK':'1'}), ('1','2843000.pbs1'))
        self.assertEqual(resolve('slurm', {'SLURM_JOB_ID':'123', 'SLURM_PROCID':'0'}), ('0','123'))
        for env in ({'PBS_JOBID':'1.pbs1'}, {'PBS_JOBID':'1.pbs1','TRIAGE_NODE_RANK':'2'}):
            with self.assertRaises(ValueError): resolve('pbs', env)

    def test_pbs_entry_rejects_one_node_before_starting_runtime(self):
        script = Path(__file__).resolve().parents[1] / 'abci_sft_multinode.pbs'
        self.assertTrue(script.exists(), 'Two-node production PBS entrypoint is missing')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);nodes=root/'nodes';nodes.write_text('node1\nnode1\n')
            env={**os.environ, 'PBS_JOBID':'1.pbs1','PBS_NODEFILE':str(nodes), 'PBS_O_WORKDIR':str(root),
                 'RTYPE':'rt_HF','ABCI_SFT_CONFIG':'run.yaml','ABCI_SFT_COMMIT':'a'*40}
            run=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('two distinct nodes',run.stderr)

    def test_small_models_use_eight_ranks(self):
        for model in ('qwen3.5-9b', 'gemma4-12b-it'):
            plan={'run':{'model':model,'execution':{'training_gpus':8,'training_devices':None,'vllm_gpu_sharing':True}},'experiment':{'attention':'sdpa'}}
            command=launch.training_command(plan,['torchrun','train_sft.py'],script='train_sft.py',master='node1',rank='0',job_id='1.pbs1',python='/env/python',scheduler='pbs')
            self.assertEqual(command[command.index('--nnodes')+1],'1')
            self.assertEqual(command[command.index('--nproc-per-node')+1],'8')
            with self.assertRaises(ValueError):
                launch.validate_layout(plan['run'])
            with self.assertRaises(ValueError):
                launch.training_command(plan,['train_sft.py'],script='train_sft.py',master='node1',rank='1',job_id='1.pbs1',python='/env/python',scheduler='pbs')

    def test_pbs_command_keeps_flex_entrypoint_and_all_training_ranks(self):
        for model,attention in [('qwen3.8-27b','sdpa'),('gemma4-31b-it','flex_attention')]:
            plan={'run':{'model':model,'execution':{'training_gpus':16,'training_devices':None,'vllm_gpu_sharing':True}},'experiment':{'attention':attention}}
            command=launch.training_command(plan,['torchrun','train_sft.py','--config','bundle.yaml'],script='train_sft.py',master='node1',rank='1',job_id='1.pbs1',python='/env/python',scheduler='pbs')
            self.assertEqual(command[command.index('--nnodes')+1],'2')
            self.assertEqual(command[command.index('--nproc-per-node')+1],'8')
            self.assertEqual(command[command.index('--node-rank')+1],'1')
            self.assertEqual(any('gemma_dynamic_flex_entry.py' in part for part in command),model.startswith('gemma'))

    def test_saved_configs_match_allocation_and_effective_batch(self):
        from scripts.train.shared.run_config import load_run, compile_run
        from scripts.train.sft.launch import resolve_training_args
        repo = Path(__file__).resolve().parents[3]
        allocation = json.loads((repo/'scripts/train/configs/abci/sft_shared_nodes.json').read_text())
        for item in allocation['runs']:
            run = load_run(repo/item['config'])
            self.assertEqual(launch.validate_layout(run, 'pbs'), item['nodes'])
            args = resolve_training_args(compile_run(run)['experiment'], item['training_gpus'])
            self.assertEqual(args['per_device_train_batch_size'] * args['gradient_accumulation_steps'] * item['training_gpus'], 16)
            self.assertEqual(item['reserved_gpus'], item['training_gpus'])
            self.assertEqual(item['training_gpus'], 8*item['nodes'])
            self.assertTrue(item['evaluation_shared'])

if __name__=='__main__':unittest.main()
