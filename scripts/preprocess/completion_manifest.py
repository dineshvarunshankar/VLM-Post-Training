"""Read the site's current completion list without trusting exported status flags."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
from threading import RLock

DEFAULT_MANIFEST = Path('00-data_website/medical/completed_clips.csv')
SOURCES = {'native': '04-segmented_snippets', 'stabilized': '05-segmented_stabilized_snippets'}
FIELDS = {'native_id', 'kind', 'level', 'stabilized_id', 'full_name',
          'native_video', 'stabilized_video', 'native_annotations', 'stabilized_annotations'}


class CompletionManifest:
    def __init__(self, path, source_root):
        self.path = Path(path)
        self.root = Path(source_root)
        self._lock = RLock()
        self._stamp = None
        self._entries = {}
        self._summary = {}
        self._refresh()

    def _refresh(self):
        with self._lock:
            stat = self.path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
            if stamp == self._stamp:
                return
            data = self.path.read_bytes()
            after = self.path.stat()
            if stamp != (after.st_mtime_ns, after.st_size, after.st_ino):
                raise ValueError('Completion manifest changed while reading; retry the run')
            reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
            if not FIELDS.issubset(reader.fieldnames or []):
                raise ValueError('Completion manifest has missing required columns')
            rows = list(reader)
            entries = {variant: [] for variant in SOURCES}
            seen = {variant: set() for variant in SOURCES}
            seen_names = {variant: set() for variant in SOURCES}
            for row in rows:
                if row['kind'] not in {'clip', 'still'} or row['level'] not in {'minimal', 'full'}:
                    raise ValueError('Completion manifest has an unknown kind or level')
                if row['kind'] == 'still':
                    continue
                for variant, source in SOURCES.items():
                    clip_id = row[f'{variant}_id']
                    name = row['full_name'] + ('_S' if variant == 'stabilized' else '')
                    if not clip_id or not clip_id.isalnum() or not name or name in {'.', '..'} or Path(name).name != name:
                        raise ValueError('Invalid clip identity in completion manifest')
                    if clip_id in seen[variant] or name in seen_names[variant]:
                        raise ValueError(f'Duplicate {variant} ID in completion manifest: {clip_id}')
                    seen[variant].add(clip_id)
                    seen_names[variant].add(name)
                    expected_video = Path(source) / (clip_id + '.mp4')
                    expected_annotations = Path(source) / '00-medical_annotations' / name
                    for field, expected in (('video', expected_video), ('annotations', expected_annotations)):
                        parts = Path(row[f'{variant}_{field}']).parts
                        if len(parts) != len(expected.parts) + 2 or parts[0] != 'datasets' or parts[1] in {'.', '..'} or parts[2:] != expected.parts:
                            raise ValueError(f'{clip_id}: manifest {variant}_{field} does not match its ID and dataset path')
                    entries[variant].append({'clip_id': clip_id, 'clip': name, 'level': row['level'],
                        'video': self.root / expected_video, 'annotations': self.root / expected_annotations})
            self._entries = entries
            self._summary = {'path': str(self.path), 'sha256': hashlib.sha256(data).hexdigest(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                'records': len(rows), 'kinds': dict(Counter(row['kind'] for row in rows)),
                'video_levels': dict(Counter(row['level'] for row in rows if row['kind'] == 'clip'))}
            self._stamp = stamp

    def entries(self, variant):
        with self._lock:
            self._refresh()
            return list(self._entries[variant])

    def summary(self):
        with self._lock:
            self._refresh()
            return dict(self._summary)

    def is_current(self, entry, variant):
        return entry in self.entries(variant)
