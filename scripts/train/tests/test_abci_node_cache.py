"""PBS scratch must survive the clean container boundary."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'abci_sft_node.sh'

class ABCINodeCacheTests(unittest.TestCase):
    def fixture(self, root):
        repo = root / 'triage/repo'
        (repo / 'scripts/train').mkdir(parents=True)
        (repo / 'scripts/train/abci_sft_node.sh').write_text(SCRIPT.read_text())
        image = root / 'image.sif'; image.touch()
        job_root = root / 'runtime/job'; job_root.mkdir(parents=True)
        local = root / 'local/123.pbs1'; local.mkdir(parents=True)
        python = root / 'env/vlm-post-training/bin/python'; python.parent.mkdir(parents=True)
        capture = root / 'inside.json'
        python.write_text('#!/usr/bin/python3\nimport json,os\nfrom pathlib import Path\n'
                          f'Path({str(capture)!r}).write_text(json.dumps(dict(os.environ)))\nraise SystemExit(77)\n')
        python.chmod(0o700)
        bins = root / 'bin'; bins.mkdir()
        args_file = root / 'singularity.json'
        singularity = bins / 'singularity'
        singularity.write_text('#!/usr/bin/python3\nimport json,os,sys\nfrom pathlib import Path\n'
                               f'Path({str(args_file)!r}).write_text(json.dumps(sys.argv[1:]))\n'
                               'i=sys.argv.index("bash")\n'
                               'os.execve("/usr/bin/bash",sys.argv[i:],{"PATH":"/usr/bin:/bin"})\n')
        singularity.chmod(0o700)
        env = {**os.environ, 'PATH':str(bins)+':'+os.environ['PATH'], 'ABCI_IMAGE':str(image),
               'PBS_LOCALDIR':str(local), 'OMPI_COMM_WORLD_RANK':'0'}
        command = ['bash',str(SCRIPT),str(repo),'config.yaml',str(job_root),'node1','123.pbs1','1']
        return env,command,local,capture,args_file

    def test_scheduler_scratch_survives_cleanenv_on_each_node(self):
        for rank in ('0','1'):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as directory:
                env,command,local,capture,args_file = self.fixture(Path(directory))
                env['OMPI_COMM_WORLD_RANK'] = rank
                result = subprocess.run(command,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,77,result.stderr)
                args = json.loads(args_file.read_text())
                self.assertIn(f'{local}:{local}',args)
                inside = json.loads(capture.read_text())
                cache = Path(inside['TMPDIR'])
                self.assertTrue(cache.is_relative_to(local),str(cache))
                self.assertTrue(cache.is_dir())
                for key in ('TRIAGE_TRITON_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','VLLM_CACHE_ROOT'):
                    self.assertTrue(Path(inside[key]).is_relative_to(cache),key)

    def test_missing_scheduler_scratch_fails_before_container(self):
        with tempfile.TemporaryDirectory() as directory:
            env,command,_,_,args_file = self.fixture(Path(directory))
            env.pop('PBS_LOCALDIR')
            result = subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('PBS_LOCALDIR',result.stderr)
            self.assertFalse(args_file.exists())

if __name__ == '__main__': unittest.main()
