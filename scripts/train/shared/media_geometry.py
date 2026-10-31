"""Verify video dimensions, decoded frame counts, FPS, duration, and writer hashes."""

import hashlib
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path


def fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def verify_video(path, expected_sha256):
    path = Path(path)
    if not isinstance(expected_sha256,str) or len(expected_sha256) != 64:
        raise ValueError('A writer video checksum is required')
    before = fingerprint(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f'Video checksum differs from writer provenance: {path}')
    result = subprocess.run(
        ['ffprobe','-threads','0','-v','error','-select_streams','v:0','-count_frames',
         '-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration',
         '-of','json',str(path)],capture_output=True,text=True,check=False,
    )
    if result.returncode or result.stderr.strip():
        raise ValueError(f'Video decode verification failed: {path}: {result.stderr[:1000]}')
    try:
        stream = json.loads(result.stdout)['streams'][0]
        geometry = dict(width=int(stream['width']),height=int(stream['height']),
            n_frames=int(stream['nb_read_frames']),fps=float(Fraction(stream['avg_frame_rate'])),
            duration_s=float(stream['duration']))
    except (KeyError,IndexError,ValueError,TypeError,ZeroDivisionError) as error:
        raise ValueError(f'Video lacks usable geometry or duration: {path}') from error
    if any(not math.isfinite(v) or v <= 0 for v in geometry.values()):
        raise ValueError(f'Video geometry must be finite and positive: {path}')
    if fingerprint(path) != before:
        raise ValueError(f'Video changed during verification: {path}')
    return geometry


def collect_geometry(rows, progress=None):
    videos = {}
    for row in rows:
        path, digest = row['video'], row.get('video_sha256')
        if path in videos and videos[path] != digest:
            raise ValueError(f'Conflicting video hashes in training examples: {path}')
        videos[path] = digest
    result = {}
    for index,(path,digest) in enumerate(videos.items(),1):
        result[path] = verify_video(path,digest)
        if progress is not None:
            progress(index,len(videos))
    return result
