"""Copy the installed Triton package to job-local storage; verify every file."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

job = os.environ.get('SLURM_JOB_ID', '')
if not job.isdigit():
    raise SystemExit('A numeric Slurm job ID is required')
source = Path(importlib.util.find_spec('triton').origin).parent.resolve()
version = importlib.metadata.version('triton')
root = Path('/tmp') / ('triage-gemma-' + job)
root.mkdir(mode=0o700, parents=True, exist_ok=True)
if root.stat().st_uid != os.getuid():
    raise SystemExit('Local cache is not owned by this user')
destination = root / 'python'
if destination.exists():
    raise SystemExit('Triton staging already exists; refusing an unverified reuse')
files = [p for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
needed = sum(p.stat().st_size for p in files)
if shutil.disk_usage(root).free < needed + 10 * 1024**3:
    raise SystemExit('Insufficient local disk space for Triton and compile caches')
staging = Path(tempfile.mkdtemp(prefix='triton-staging-', dir=root))
manifest = {'version': version, 'source': str(source), 'files': {}}
try:
    for src in files:
        relative = src.relative_to(source)
        target = staging / 'triton' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = src.stat()
        digest = hashlib.sha256()
        with src.open('rb') as reader, target.open('wb') as writer:
            while block := reader.read(4 * 1024**2):
                digest.update(block)
                writer.write(block)
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Source changed during copy: {relative}')
        with target.open('rb') as reader:
            verified = hashlib.file_digest(reader, 'sha256').hexdigest()
        if verified != digest.hexdigest():
            raise RuntimeError(f'Local checksum mismatch: {relative}')
        shutil.copymode(src, target)
        manifest['files'][str(relative)] = verified
    (staging / 'triton-stage-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    staging.rename(destination)
except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise
print(json.dumps({'triton_version': version, 'files': len(files), 'bytes': needed, 'python_path': str(destination)}), flush=True)
