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
                existing = owner.objects.get(self.path)
                if self.headers.get('If-None-Match') == '*' and existing:
                    self.send_error(412)
                    return
                if self.headers.get('If-Match') and (not existing or hashlib.md5(existing[0]).hexdigest() != self.headers['If-Match']):
                    self.send_error(412)
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

    def upload(self, **kwargs):
        return self.module().upload_dataset(self.plan, 'swift://bucket/snapshot', workers=2, source=self.source, **kwargs)

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

    def test_update_replaces_changed_file_and_preserves_unrelated(self):
        self.upload()
        self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        unrelated = self.dest / 'keep.txt'
        unrelated.write_text('keep')
        p = self.root / self.name
        p.write_bytes(b'newer-payload')
        self.plan.stamps[p] = transfer.stamp(p)
        self.upload()
        self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        self.assertEqual((self.dest / self.name).read_bytes(), b'newer-payload')
        self.assertEqual(unrelated.read_text(), 'keep')

    def test_rewrite_download_replaces_even_matching_file(self):
        self.upload()
        self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        path = self.dest / self.name
        inode = path.stat().st_ino
        self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source, rewrite=True)
        self.assertNotEqual(path.stat().st_ino, inode)
        self.assertEqual(path.read_bytes(), b'video-payload')

    def test_failed_rewrite_keeps_existing_file_and_receipt(self):
        self.upload()
        self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        receipt = (self.dest / 'dataset-transfer.json').read_bytes()
        key = '/bucket/snapshot/files/' + self.name
        _, headers = self.objects[key]
        self.objects[key] = (b'wrong-payload', headers)
        with self.assertRaises(ValueError):
            self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source, rewrite=True)
        self.assertEqual((self.dest / self.name).read_bytes(), b'video-payload')
        self.assertEqual((self.dest / 'dataset-transfer.json').read_bytes(), receipt)

    def test_update_skips_matching_upload_but_rewrite_uploads(self):
        self.upload()
        self.fail_put = True
        self.upload()
        with self.assertRaises(ValueError):
            self.upload(rewrite=True)

    def test_failed_update_preserves_published_manifest_and_objects(self):
        self.upload()
        before = dict(self.objects)
        p = self.root / self.name
        p.write_bytes(b'newer-payload')
        self.plan.stamps[p] = transfer.stamp(p)
        self.fail_put = True
        with self.assertRaises(ValueError):
            self.upload()
        self.assertEqual(self.objects, before)

    def test_old_manifest_objects_remain_readable_after_update(self):
        self.upload()
        legacy_key = '/bucket/snapshot/files/' + self.name
        original = self.objects[legacy_key]
        p = self.root / self.name
        p.write_bytes(b'newer-payload')
        self.plan.stamps[p] = transfer.stamp(p)
        manifest = self.upload()
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(self.objects[legacy_key], original)
        self.assertEqual(self.objects['/bucket/snapshot/' + manifest['files'][self.name]['object']][0], b'newer-payload')

    def test_symlink_destination_rejected(self):
        self.upload()
        other = Path(self.tmp.name) / 'other'
        other.mkdir()
        self.dest.mkdir()
        (self.dest / '06-native_media_dataset').symlink_to(other, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        self.assertEqual(list(other.iterdir()), [])

    def test_medical_scan_ignores_other_gt_writes_but_guards_medical_gt(self):
        gt = self.root / 'GT'
        gt.mkdir()
        medical = gt / 'medical_vqa.json'
        medical.write_text('{}')
        (gt / 'general_scene_description.json').write_text('{}')
        files, stamps = transfer.scan_directory(self.root, medical_gt_only=True)
        self.assertIn(medical, files)
        self.assertNotIn(gt / 'general_scene_description.json', files)
        plan = transfer.TransferPlan(self.root, [], stamps, {}, 0)
        (gt / 'general_scene_description.lock').touch()
        plan.verify_unchanged()
        medical.write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            plan.verify_unchanged()

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

    def test_subject_media_cloud_roundtrip_requires_explicit_opt_in(self):
        name = '10-other_datasets/people_negatives_VK_20261102/rgb/box/image.jpg'
        path = self.root / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b'subject-image')
        self.plan.files.append(name)
        self.plan.stamps[path] = transfer.stamp(path)
        with self.assertRaises(ValueError):
            self.upload()
        self.upload(include_subject_check=True)
        with self.assertRaises(ValueError):
            self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source)
        self.assertFalse(self.dest.exists())
        self.module().download_dataset('swift://bucket/snapshot', self.dest, source=self.source, include_subject_check=True)
        self.assertEqual((self.dest / name).read_bytes(), b'subject-image')
