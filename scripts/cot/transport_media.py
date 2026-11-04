"""Validated, disposable API video copies; prepared dataset media remain unchanged."""
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from threading import RLock

PROFILE = {'version': 1, 'codec': 'libx264', 'crf': 18, 'preset': 'medium',
           'frames': 'passthrough', 'resolution': 'source', 'timestamps': 'source'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(path):
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-threads', '0', '-select_streams', 'v:0',
        '-show_streams', '-show_frames', '-show_entries',
        'stream=width,height,pix_fmt,sample_aspect_ratio,color_range,color_space,color_transfer,color_primaries,time_base,avg_frame_rate:'
        'frame=best_effort_timestamp_time', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True, timeout=300)
    data = json.loads(result.stdout)
    if len(data.get('streams', [])) != 1 or not data.get('frames'):
        raise ValueError('Transport video must contain a decoded video stream')
    return {'stream': data['streams'][0],
            'times': [row['best_effort_timestamp_time'] for row in data['frames']]}


def validate_timing(source, encoded):
    # MP4 average rates can differ by rounding of the final packet duration.
    # Individual presentation timestamps and frame counts remain authoritative.
    a, b = source['stream'], encoded['stream']
    rate_equal = abs(float(Fraction(a['avg_frame_rate'])) - float(Fraction(b['avg_frame_rate']))) <= 0.00001
    if not rate_equal or {k: v for k, v in a.items() if k != 'avg_frame_rate'} != {k: v for k, v in b.items() if k != 'avg_frame_rate'}:
        raise ValueError('Transport encoding changed video geometry, FPS, pixel format, or color metadata')
    if len(source['times']) != len(encoded['times']) or any(
            abs(Decimal(a) - Decimal(b)) > Decimal('0.000001')
            for a, b in zip(source['times'], encoded['times'])):
        raise ValueError('Transport encoding changed frame count or timestamps')


@dataclass(frozen=True)
class TransportVideo:
    path: Path
    metadata: dict


class TransportCache:
    def __init__(self, directory):
        self.directory = Path(directory).resolve() / 'transport-v1'
        self._mutex = RLock()
        self._pending = {}
        self._failed = set()
        self._assets = {}

    @contextmanager
    def _lock(self, key):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / (key + '.lock')).open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield
            # Keep the lock inode: unlinking it could split concurrent lock owners.

    def register(self, clip, tasks):
        with self._mutex:
            self._pending[str(Path(clip).resolve())] = set(tasks)

    def get(self, source, source_sha256):
        source = Path(source)
        if sha256(source) != source_sha256:
            raise ValueError('Source video changed before transport encoding')
        key = hashlib.sha256(json.dumps({'source': source_sha256, 'profile': PROFILE},
                                       sort_keys=True).encode()).hexdigest()
        path = self.directory / (key + '.mp4')
        record = self.directory / (key + '.json')
        with self._lock(key):
            metadata = None
            try:
                candidate = json.loads(record.read_text())
                if (candidate['source_sha256'] == source_sha256 and candidate['profile'] == PROFILE
                        and path.is_file() and sha256(path) == candidate['sha256']):
                    metadata = candidate
            except (OSError, ValueError, KeyError):
                pass
            if metadata is None:
                original = probe(source)
                with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.mp4', delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-copyts',
                               '-i', str(source), '-map', '0:v:0', '-map', '0:a?',
                               '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-threads', '2',
                               '-pix_fmt', original['stream']['pix_fmt'], '-fps_mode', 'passthrough',
                               '-enc_time_base', 'demux', '-c:a', 'copy', '-map_metadata', '0',
                               '-avoid_negative_ts', 'disabled', '-movflags', '+faststart']
                    for field, option in [('color_range', '-color_range'), ('color_space', '-colorspace'),
                                          ('color_transfer', '-color_trc'), ('color_primaries', '-color_primaries')]:
                        value = original['stream'].get(field)
                        if value and value != 'unknown':
                            command.extend([option, value])
                    subprocess.run(command + [str(temporary)], check=True, capture_output=True, timeout=900)
                    validate_timing(original, probe(temporary))
                    if sha256(source) != source_sha256:
                        raise ValueError('Source video changed during transport encoding')
                    metadata = {'source_sha256': source_sha256, 'source_bytes': source.stat().st_size,
                                'sha256': sha256(temporary), 'bytes': temporary.stat().st_size,
                                'profile': dict(PROFILE), 'frames': len(original['times']),
                                'encoder': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]}
                    temporary.replace(path)
                    with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as stream:
                        record_temp = Path(stream.name)
                        json.dump(metadata, stream)
                    record_temp.replace(record)
                finally:
                    temporary.unlink(missing_ok=True)
            with self._mutex:
                self._assets.setdefault(str(source.parent.resolve()), set()).add(key)
            return TransportVideo(path, metadata)

    def finish(self, clip, task, success):
        clip = str(Path(clip).resolve())
        with self._mutex:
            if clip not in self._pending:
                return
            if not success:
                self._failed.add(clip)
            self._pending[clip].discard(task)
            if self._pending[clip] or clip in self._failed:
                return
            keys = self._assets.pop(clip, set())
            # Identical source bytes may be shared by another selected clip.
            others = set().union(*self._assets.values()) if self._assets else set()
            keys -= others
        for key in keys:
            with self._lock(key):
                (self.directory / (key + '.mp4')).unlink(missing_ok=True)
                (self.directory / (key + '.json')).unlink(missing_ok=True)


_CACHES = {}
_CACHE_LOCK = RLock()


def transport_cache(directory):
    key = str(Path(directory).resolve())
    with _CACHE_LOCK:
        if key not in _CACHES:
            _CACHES[key] = TransportCache(directory)
        return _CACHES[key]
