"""Reuse source data within one clip; reject filesystem changes before publication."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

_active = ContextVar('cot_clip_sources', default=None)


def _stamp(path):
    try:
        link, target = path.lstat(), path.stat()
    except FileNotFoundError:
        return None
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    entries = tuple(sorted(p.name for p in path.iterdir())) if path.is_dir() else None
    return (str(path.resolve()), fields(link), fields(target), entries)


class _ClipSources:
    def __init__(self):
        self.stamps = {}
        self.values = {}

    def watch(self, paths):
        for path in paths:
            current = _stamp(path)
            if path in self.stamps and current != self.stamps[path]:
                raise ValueError(f'Source changed during clip processing: {path}')
            self.stamps[path] = current

    def get(self, paths, key, loader):
        self.watch(paths)
        identity = (paths, key)
        if identity not in self.values:
            value = loader()
            self.watch(paths)
            self.values[identity] = value
        return self.values[identity]


@contextmanager
def clip_sources():
    """Each worker releases its cached bytes and decoded frames after this clip."""
    token = _active.set(_ClipSources())
    try:
        yield
    finally:
        _active.reset(token)


def watch_sources(paths):
    cache = _active.get()
    if cache is not None:
        cache.watch(tuple(Path(p).absolute() for p in paths))


def verify_sources():
    cache = _active.get()
    if cache is not None:
        cache.watch(tuple(cache.stamps))
        # Small annotation/config files get a content check even when timestamps
        # have coarse resolution. Large MP4s use their recorded file identity.
        for (paths, key), value in cache.values.items():
            if key == 'bytes' and len(paths) == 1 and paths[0].suffix.lower() != '.mp4':
                if paths[0].read_bytes() != value:
                    raise ValueError(f'Source changed during clip processing: {paths[0]}')


def caching_sources():
    return _active.get() is not None


def cached_files(paths, key, loader):
    cache = _active.get()
    if cache is None:
        return loader()
    return cache.get(tuple(Path(p).absolute() for p in paths), key, loader)


def file_bytes(path):
    path = Path(path)
    return cached_files([path], 'bytes', path.read_bytes)


def digest(path):
    return cached_files([path], 'sha256', lambda: hashlib.sha256(file_bytes(path)).hexdigest())


def read_json(path):
    return deepcopy(cached_files([path], 'json', lambda: json.loads(file_bytes(path))))


def read_yaml(path):
    return deepcopy(cached_files([path], 'yaml', lambda: yaml.safe_load(file_bytes(path))))
