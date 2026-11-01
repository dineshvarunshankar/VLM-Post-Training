import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from scripts.train import transfer_dataset as transfer
from scripts.train.storage.cloud_models import SwiftSource


class CloudDatasetTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        self.root.mkdir()
        self.name = '06-native_media_dataset/clip/clip.mp4'
        p = self.root / self.name
        p.parent.mkdir(parents=True)
        p.write_bytes(b'video-payload')
        self.plan = transfer.TransferPlan(self.root, [self.name], {p: transfer.stamp(p)}, {'native': 1}, 13)
        self.objects = {}
        self.fail_put = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_PUT(self):
                if owner.fail_put:
                    self.send_error(400)
                    return
                data = self.rfile.read(int(self.headers['Content-Length']))
                md5 = hashlib.md5(data).hexdigest()
                if self.headers.get('ETag', md5) != md5:
                    self.send_error(422)
                    return
                owner.objects[self.path] = (data, dict(self.headers))
                self.send_response(201)
                self.send_header('Content-Length', '0')
                self.end_headers()

            def serve(self, head=False):
                if self.path not in owner.objects:
                    self.send_error(404)
                    return
                data, headers = owner.objects[self.path]
                offset = int(self.headers['Range'].split('=')[1].split('-')[0]) if 'Range' in self.headers else 0
                self.send_response(206 if offset else 200)
                if offset:
                    self.send_header('Content-Range', f'bytes {offset}-{len(data)-1}/{len(data)}')
                self.send_header('Content-Length', str(len(data)-offset))
                for k, v in headers.items():
                    if k.lower().startswith('x-object-meta-'):
                        self.send_header(k, v)
                self.end_headers()
                if not head:
                    self.wfile.write(data[offset:])

            def do_HEAD(self):
                self.serve(True)

            def do_GET(self):
                self.serve()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.source = SwiftSource(f'http://127.0.0.1:{self.server.server_port}')
        self.dest = Path(self.tmp.name) / 'destination'

    def module(self):
        from scripts.train.storage import dataset_transfer
        return dataset_transfer

    def upload(self):
        return self.module().upload_dataset(self.plan, 'swift://bucket/snapshot', workers=2, source=self.source)

    def test_roundtrip_preserves_layout_and_resumes_partial(self):
        self.upload()
        partial = self.dest / (self.name + '.part')
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b'video-')
        self.module().download_dataset('swift://bucket/snapshot', self.dest, workers=2, source=self.source)
        self.assertEqual((self.dest / self.name).read_bytes(), b'video-payload')
        self.assertFalse(partial.exists())
        self.assertTrue((self.dest / 'dataset-transfer.json').is_file())
        # A rerun verifies and keeps completed files.
        self.module().download_dataset('swift://bucket/snapshot', self.dest, workers=2, source=self.source)

    def test_corrupt_download_is_not_published(self):
        self.upload()
        key = '/bucket/snapshot/files/' + self.name
        _, headers = self.objects[key]
        self.objects[key] = (b'wrong-payload', headers)
        with self.assertRaises(ValueError):
            self.module().download_dataset('swift://bucket/snapshot', self.dest, workers=2, source=self.source)
        self.assertFalse((self.dest / self.name).exists())
        self.assertFalse((self.dest / 'dataset-transfer.json').exists())

    def test_unsafe_manifest_is_rejected_before_writing(self):
        manifest = self.upload()
        record = manifest['files'].pop(self.name)
        manifest['files']['../outside'] = record
        self.objects['/bucket/snapshot/dataset-transfer.json'] = (json.dumps(manifest).encode(), {})
        with self.assertRaises(ValueError):
            self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        self.assertFalse(self.dest.exists())

    def test_failed_upload_does_not_publish_manifest(self):
        self.fail_put = True
        with self.assertRaises(Exception):
            self.upload()
        self.assertNotIn('/bucket/snapshot/dataset-transfer.json', self.objects)

    def test_source_change_prevents_upload(self):
        (self.root / self.name).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.upload()
        self.assertEqual(self.objects, {})

    def test_conflicting_snapshot_is_not_overwritten(self):
        self.upload()
        before = dict(self.objects)
        p = self.root / self.name
        p.write_bytes(b'newer-payload')
        self.plan.stamps[p] = transfer.stamp(p)
        with self.assertRaises(ValueError):
            self.upload()
        self.assertEqual(before, self.objects)

    def test_symlink_destination_rejected(self):
        self.upload()
        other = Path(self.tmp.name) / 'other'
        other.mkdir()
        self.dest.mkdir()
        (self.dest / '06-native_media_dataset').symlink_to(other, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        self.assertEqual(list(other.iterdir()), [])

    def test_cli_download_and_preview(self):
        import os
        from unittest.mock import patch
        self.upload()
        with patch.dict(os.environ, {'AIRLAB_SWIFT_URL': self.source.url}):
            args = ['--cloud-source', 'swift://bucket/snapshot', '--destination', str(self.dest)]
            self.assertEqual(transfer.main(args + ['--inspect']), 0)
            self.assertFalse(self.dest.exists())
            self.assertEqual(transfer.main(args + ['--workers', '2']), 0)
        self.assertEqual((self.dest / self.name).read_bytes(), b'video-payload')
