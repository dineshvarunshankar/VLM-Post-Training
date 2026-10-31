"""Publish and read the eligible inventory of prepared datasets, independent of the site."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

DATASETS = {'native': '06-native_media_dataset', 'stabilized': '07-stabilized_media_dataset'}
MANIFEST_NAME = 'dataset_manifest.json'
SUCCESS = {'written', 'rewritten', 'unchanged'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class PreparedManifest:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes, self._entries, self._summary = {}, {}, {}
        for variant, name in DATASETS.items():
            path = self.root / name / MANIFEST_NAME
            if not path.is_file():
                raise ValueError(f'Missing prepared dataset manifest: {path}. Run completed preprocessing for both variants first.')
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError(f'Unsafe prepared manifest path: {path}')
            data = path.read_bytes()
            document = json.loads(data)
            if not isinstance(document, dict) or document.get('schema_version') != 1 or document.get('variant') != variant:
                raise ValueError(f'Invalid prepared dataset manifest: {path}')
            entries = document.get('clips')
            if not isinstance(entries, list):
                raise ValueError(f'Missing clips in prepared dataset manifest: {path}')
            ids, names = set(), set()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(f'Invalid clip record in prepared manifest: {path}')
                clip_id, clip = entry.get('clip_id', ''), entry.get('clip', '')
                digest = entry.get('provenance_sha256', '')
                if (not isinstance(clip_id, str) or not clip_id.isalnum() or
                        not isinstance(clip, str) or not clip or Path(clip).name != clip or clip in {'.', '..'} or
                        clip_id in ids or clip in names or entry.get('level') not in {'minimal', 'full'} or
                        not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError(f'Invalid or duplicate clip in prepared dataset manifest: {path}')
                ids.add(clip_id)
                names.add(clip)
            self.hashes[path] = sha256(data)
            self._entries[variant] = entries
            self._summary[variant] = {'path': str(path), 'sha256': self.hashes[path], 'clips': len(entries),
                                      'created_at': document.get('created_at')}

    def entries(self, variant):
        return list(self._entries[variant])

    def summary(self):
        return dict(self._summary)

    def verify_unchanged(self):
        for path, expected in self.hashes.items():
            if sha256(path.read_bytes()) != expected:
                raise ValueError('Prepared dataset manifest changed during preparation; retry after preprocessing finishes')


def publish_manifests(report, completion, root, variants):
    """Index successful outputs only after a full completed-selection run."""
    if (completion is None or report.get('selection') != 'completed' or
            report.get('interrupted') or report.get('limit') is not None):
        return {}
    root = Path(root).resolve()
    completion_snapshot = completion.summary()
    published = {}
    for variant in variants:
        eligible = {entry['clip']: entry for entry in completion.entries(variant)}
        clips = []
        for record in report['clips']:
            if record['variant'] != variant or record['status'] not in SUCCESS:
                continue
            entry = eligible.get(record['clip'])
            if entry is None:
                continue
            destination = root / DATASETS[variant] / entry['clip_id'] / (entry['clip_id'] + '.mp4')
            if Path(record['destination']) != destination:
                continue
            provenance = destination.parent / 'provenance.json'
            if (destination.is_symlink() or provenance.is_symlink() or
                    not destination.resolve().is_relative_to(root) or not destination.is_file()):
                raise ValueError(f'Unsafe or missing prepared output: {destination}')
            data = provenance.read_bytes()
            if json.loads(data).get('clip') != entry['clip']:
                raise ValueError(f'Prepared provenance identity differs: {provenance}')
            clips.append({'clip_id': entry['clip_id'], 'clip': entry['clip'], 'level': entry['level'],
                          'modality': record.get('modality'), 'provenance_sha256': sha256(data)})
        document = {'schema_version': 1, 'variant': variant,
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'run_started_at': report['created_at'], 'source_completion': completion_snapshot,
                    'clips': sorted(clips, key=lambda c: c['clip_id'])}
        folder = root / DATASETS[variant]
        folder.mkdir(parents=True, exist_ok=True)
        if folder.is_symlink():
            raise ValueError(f'Unsafe dataset directory: {folder}')
        path = folder / MANIFEST_NAME
        lock = folder / '.dataset-manifest.lock'
        lock.mkdir()
        temporary = None
        try:
            if path.is_symlink():
                raise ValueError(f'Unsafe dataset manifest: {path}')
            if path.exists() and json.loads(path.read_text()).get('run_started_at', '') > report['created_at']:
                raise ValueError('A newer preprocessing run already published the dataset manifest')
            if completion.summary()['sha256'] != completion_snapshot['sha256']:
                raise ValueError('Completion list changed during manifest publication; rerun preprocessing')
            with tempfile.NamedTemporaryFile(mode='w', dir=folder, prefix='.dataset-manifest-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(document, stream, indent=2)
                stream.write('\n')
            temporary.replace(path)
            published[variant] = {'path': str(path), 'clips': len(clips)}
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            lock.rmdir()
    return published
