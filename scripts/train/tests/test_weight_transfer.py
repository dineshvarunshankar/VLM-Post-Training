import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.train import transfer_dataset as transfer

class WeightTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        self.model = self.root / 'org/model/revision'
        self.model.mkdir(parents=True)
        (self.model / 'config.json').write_bytes(b'{}')
        (self.model / 'weights.safetensors').write_bytes(b'weights')
        self.records = {n: {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
                        for n, b in [('config.json', b'{}'), ('weights.safetensors', b'weights')]}
        self.save_manifest()

    def save_manifest(self):
        (self.model / 'cloud_source.json').write_text(json.dumps(dict(
            repository='org/model', revision='revision', files=self.records)))

    def test_valid_and_corrupt_source(self):
        plan = transfer.prepare_weights(self.root)
        self.assertEqual(len(plan.files), 3)
        (self.model / 'weights.safetensors').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            transfer.prepare_weights(self.root)

    def test_unsafe_manifest(self):
        self.records['../escape'] = self.records['config.json']
        self.save_manifest()
        with self.assertRaises(ValueError):
            transfer.prepare_weights(self.root)

    def test_symlink_rejected(self):
        p = self.model / 'weights.safetensors'
        p.unlink()
        p.symlink_to(self.model / 'config.json')
        with self.assertRaises(ValueError):
            transfer.prepare_weights(self.root)

    def test_destination_corruption(self):
        plan = transfer.prepare_weights(self.root)
        (self.model / 'weights.safetensors').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            transfer.verify_weights_destination(plan, str(self.root))

    def test_failed_copy_does_not_verify_or_succeed(self):
        plan = transfer.prepare_weights(self.root)
        with patch.object(transfer.shutil, 'which', return_value='/usr/bin/rsync'), \
             patch.object(transfer.subprocess, 'run') as run, \
             patch.object(transfer, 'verify_weights_destination') as verify:
            run.return_value.returncode = 23
            self.assertEqual(transfer.transfer(plan, str(Path(self.tmp.name)/'dest')), 23)
            self.assertEqual(verify.call_count, 1)
            self.assertTrue(verify.call_args.kwargs['preflight'])

    def test_real_rsync_repairs_same_size_corruption_and_preserves_unrelated(self):
        import shutil
        if not shutil.which('rsync'):
            self.skipTest('rsync unavailable')
        plan = transfer.prepare_weights(self.root)
        dest = Path(self.tmp.name) / 'dest'
        target = dest / 'org/model/revision'
        target.mkdir(parents=True)
        (target / 'weights.safetensors').write_bytes(b'corrupt')
        (dest / 'unrelated.txt').write_text('keep')
        self.assertEqual(transfer.transfer(plan, str(dest), inspect=True), 0)
        self.assertEqual((target / 'weights.safetensors').read_bytes(), b'corrupt')
        self.assertEqual(transfer.transfer(plan, str(dest)), 0)
        self.assertEqual((target / 'weights.safetensors').read_bytes(), b'weights')
        self.assertEqual((dest / 'unrelated.txt').read_text(), 'keep')

    def test_destination_symlink_is_rejected_before_copy(self):
        plan = transfer.prepare_weights(self.root)
        dest = Path(self.tmp.name) / 'dest'
        dest.mkdir()
        (dest / 'org').symlink_to(self.root / 'org', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            transfer.verify_weights_destination(plan, str(dest), preflight=True)

    def test_preview_does_not_read_weight_contents(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == 'weights.safetensors':
                self.fail('Preview must not read weight contents')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            plan = transfer.prepare_weights(self.root, check_hashes=False)
        self.assertEqual(len(plan.files), 3)
        (self.model / 'weights.safetensors').write_bytes(b'short')
        with self.assertRaisesRegex(ValueError, 'size'):
            transfer.prepare_weights(self.root, check_hashes=False)

    def test_preview_rsync_does_not_request_checksum_scan(self):
        plan = transfer.prepare_weights(self.root)
        with patch.object(transfer.shutil, 'which', return_value='/usr/bin/rsync'), \
             patch.object(transfer.subprocess, 'run') as run:
            run.return_value.returncode = 0
            transfer.transfer(plan, str(Path(self.tmp.name)/'dest'), inspect=True)
            self.assertNotIn('--checksum', run.call_args.args[0])
            self.assertIn('--dry-run', run.call_args.args[0])

    def test_hashing_reports_progress(self):
        import contextlib
        import io
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            transfer.prepare_weights(self.root)
        self.assertIn('SHA-256', output.getvalue())
        self.assertIn('weights.safetensors', output.getvalue())

    def test_actual_transfer_does_not_hash_source_in_python(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path == self.model / 'weights.safetensors':
                self.fail('Source must be read by rsync only')
            return original(path, *args, **kwargs)
        dest = Path(self.tmp.name) / 'destination'
        with patch.object(Path, 'open', guarded):
            rc = transfer.main(['--weights-root', str(self.root), '--destination', str(dest)])
        self.assertEqual(rc, 0)
        self.assertEqual((dest / 'org/model/revision/weights.safetensors').read_bytes(), b'weights')

    def test_corrupt_source_cannot_report_transfer_success(self):
        (self.model / 'weights.safetensors').write_bytes(b'corrupt')
        dest = Path(self.tmp.name) / 'destination'
        rc = transfer.main(['--weights-root', str(self.root), '--destination', str(dest)])
        self.assertEqual(rc, 1)

    def test_real_transfer_avoids_rsync_prechecksum_pass(self):
        plan = transfer.prepare_weights(self.root, check_hashes=False)
        with patch.object(transfer.shutil, 'which', return_value='/usr/bin/rsync'), \
             patch.object(transfer.subprocess, 'run') as run, \
             patch.object(transfer, 'verify_weights_destination'):
            run.return_value.returncode = 0
            transfer.transfer(plan, str(Path(self.tmp.name)/'dest'))
            self.assertNotIn('--checksum', run.call_args.args[0])
            self.assertIn('--ignore-times', run.call_args.args[0])

    def test_stale_adapter_rejected(self):
        plan = transfer.prepare_weights(self.root)
        (self.model / 'adapter_config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            transfer.verify_weights_destination(plan, str(self.root), preflight=True)

    def test_remote_verifier_protocol(self):
        import subprocess
        plan = transfer.prepare_weights(self.root)
        real_run = subprocess.run
        def local_remote(command, **kwargs):
            import shlex
            self.assertEqual(command[:2], ['ssh', 'test-host'])
            return real_run(shlex.split(command[2]), **kwargs)
        with patch.object(transfer.subprocess, 'run', side_effect=local_remote):
            transfer.verify_weights_destination(plan, 'test-host:' + str(self.root))

if __name__ == '__main__':
    unittest.main()
