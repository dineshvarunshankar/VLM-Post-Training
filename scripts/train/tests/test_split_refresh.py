"""Split replacement safeguards, exercised only in temporary directories."""
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.train.splits.cli import arguments, publish_split


class SplitRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.destination = self.root / '08-train_val_test/2026-10-31/answer_only'
        self.report = {'input_fingerprint_sha256': 'original', 'policy': {'seed': 42}}
        self.selection = self.make_selection('old')
        publish_split(self.destination, self.selection, self.report)

    @staticmethod
    def make_selection(prefix):
        return {name: [SimpleNamespace(rows=[{'variant': 'native', 'clip': f'{prefix}-{name}',
                                             'category': 'severe_hemorrhage'}])]
                for name in ('train', 'validation', 'test')}

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.destination.iterdir()}

    def test_existing_split_requires_refresh(self):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            publish_split(self.destination, self.make_selection('new'), self.report)
        self.assertEqual(self.snapshot(), before)

    def test_refresh_replaces_all_files_and_keeps_no_archive(self):
        status = publish_split(self.destination, self.make_selection('new'),
                               {**self.report, 'input_fingerprint_sha256': 'changed'}, refresh=True)
        self.assertEqual(status, 'replaced')
        summary = json.loads((self.destination / 'summary.json').read_text())
        for name in ('train', 'validation', 'test'):
            data = (self.destination / f'{name}.jsonl').read_bytes()
            self.assertEqual(json.loads(data)['clip'], f'new-{name}')
            self.assertEqual(summary['split_sha256'][name], hashlib.sha256(data).hexdigest())
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_identical_refresh_does_not_rewrite(self):
        before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
                  for p in self.destination.iterdir()}
        self.assertEqual(publish_split(self.destination, self.selection, self.report, refresh=True), 'unchanged')
        after = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
                 for p in self.destination.iterdir()}
        self.assertEqual(after, before)

    def test_failed_preparation_keeps_existing_split(self):
        before = self.snapshot()
        with self.assertRaises(ValueError):
            publish_split(self.destination, self.make_selection('new'), {'invalid': float('nan')}, refresh=True)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_failed_install_restores_existing_split(self):
        before = self.snapshot()
        rename = Path.rename
        def fail_new_install(source, target):
            if source.name == 'split' and Path(target) == self.destination:
                raise OSError('Simulated installation failure')
            return rename(source, target)
        with patch.object(Path, 'rename', fail_new_install), self.assertRaises(OSError):
            publish_split(self.destination, self.make_selection('new'), self.report, refresh=True)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])

    def test_refresh_rejects_unexpected_files(self):
        (self.destination / 'notes.txt').write_text('User content')
        before = self.snapshot()
        with self.assertRaises(ValueError):
            publish_split(self.destination, self.make_selection('new'), self.report, refresh=True)
        self.assertEqual(self.snapshot(), before)

    def test_refresh_rejects_symlink_destination(self):
        link = self.destination.parent / 'linked'
        link.symlink_to(self.destination, target_is_directory=True)
        before = self.snapshot()
        with self.assertRaises(ValueError):
            publish_split(link, self.make_selection('new'), self.report, refresh=True)
        self.assertEqual(self.snapshot(), before)

    def test_refresh_creates_missing_split(self):
        destination = self.root / 'new-date/answer_only'
        self.assertEqual(publish_split(destination, self.selection, self.report, refresh=True), 'created')
        self.assertTrue((destination / 'summary.json').is_file())

    def test_cli_accepts_refresh_and_inspect_without_touching_split(self):
        options = ['--media-root', str(self.root), '--modality', 'RGB',
                   '--percentage', '5', '--date', '2026-10-31']
        before = self.snapshot()
        with patch('scripts.train.splits.cli.load_dotenv'), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                arguments(options)
            self.assertEqual(error.exception.code, 2)
            self.assertTrue(arguments([*options, '--refresh']).refresh)
            self.assertTrue(arguments([*options, '--refresh', '--inspect']).inspect)
        self.assertEqual(self.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
