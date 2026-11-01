#!/usr/bin/env python3
"""Preview or copy prepared datasets or verified base weights over rsync/SSH.

Weights: --weights-root PATH --destination SSH_ALIAS:/absolute/base-weights --inspect
The SSH alias must work from this machine. Both ends need rsync; the remote
weight verifier also needs python3. Preview checks metadata only, not content
checksums. Remove --inspect to copy and verify hashes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import inspect as source_inspect
import shlex
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocess.prepared_manifest import DATASETS, PreparedManifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


@dataclass
class TransferPlan:
    root: Path
    files: list[str]
    stamps: dict[Path, tuple]
    counts: dict[str, int]
    total_bytes: int
    checksums: dict | None = None

    def verify_unchanged(self):
        def check(item):
            path, expected = item
            if path.is_symlink() or stamp(path) != expected:
                raise ValueError(f'Source changed; stop preprocessing and retry: {path}')
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check, self.stamps.items()))


def scan_directory(directory):
    files, stamps = [], {}

    def visit(path):
        if path.is_symlink():
            raise ValueError(f'Symlinks are not supported in transfer inputs: {path}')
        stamps[path] = stamp(path)
        with os.scandir(path) as entries:
            for entry in entries:
                child = Path(entry.path)
                if entry.is_symlink():
                    raise ValueError(f'Symlinks are not supported in transfer inputs: {child}')
                if entry.is_dir():
                    visit(child)
                elif entry.is_file():
                    files.append(child)
                    stamps[child] = stamp(child)
                else:
                    raise ValueError(f'Unsupported source file: {child}')
    visit(directory)
    return files, stamps


def prepare_transfer(root, split):
    root, split = Path(root).resolve(), Path(split).absolute()
    if split.is_symlink() or not split.resolve().is_relative_to(root / '08-train_val_test'):
        raise ValueError('--split-dir must be inside media-root/08-train_val_test')
    split = split.resolve()
    manifest = PreparedManifest(root)
    files, stamps = scan_directory(split)
    summary_path = split / 'summary.json'
    report = json.loads(summary_path.read_text())
    ids = {}
    directories = []
    counts = {}
    for variant, folder in DATASETS.items():
        path = root / folder / 'dataset_manifest.json'
        if report['prepared_manifests'][variant]['sha256'] != manifest.hashes[path]:
            raise ValueError(f'Split uses a different {variant} manifest; choose a matching dataset snapshot')
        files.append(path)
        stamps[path] = stamp(path)
        entries = manifest.entries(variant)
        counts[variant] = len(entries)
        ids[variant] = {entry['clip_id'] for entry in entries}
        for entry in entries:
            directories.append((root / folder / entry['clip_id'], entry))

    def check_clip(item):
        directory, entry = item
        if directory.is_symlink() or not directory.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe clip directory: {directory}')
        required = [directory / (entry['clip_id'] + suffix) for suffix in ('.mp4', '.json')]
        required += [directory / 'all_annotations/meta.json', directory / 'provenance.json']
        for path in required:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Missing or unsafe prepared file: {path}')
        if digest(directory / 'provenance.json') != entry['provenance_sha256']:
            raise ValueError(f'Prepared provenance changed: {directory}')
        return scan_directory(directory)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for clip_files, clip_stamps in pool.map(check_clip, directories):
            files.extend(clip_files)
            stamps.update(clip_stamps)
    for name in ('train', 'validation', 'test'):
        path = split / f'{name}.jsonl'
        if digest(path) != report['split_sha256'][name]:
            raise ValueError(f'Split checksum mismatch: {path}')
        for line in path.read_text().splitlines():
            row = json.loads(line)
            variant, clip = row['variant'], row['clip']
            if clip not in ids.get(variant, set()):
                raise ValueError(f'Split references an unpublished clip: {variant}/{clip}')
            gt = root / DATASETS[variant] / clip / 'GT/medical_vqa.json'
            if gt not in stamps:
                raise ValueError(f'Missing GT for split clip: {gt}')
    manifest.verify_unchanged()
    files = sorted(set(files))
    return TransferPlan(root, [str(path.relative_to(root)) for path in files], stamps,
                        counts, sum(stamps[path][0] for path in files))


def check_weight_files(root, records, preflight=False, check_hashes=True):
    """Standalone verifier, also executed over SSH using only Python's standard library."""
    from pathlib import Path
    import hashlib
    import time
    root = Path(root)
    for name in records:
        if name.endswith('/cloud_source.json'):
            directory = root / Path(name).parent
            if directory.is_symlink():
                raise ValueError(f'Symlink in weight path: {directory}')
            if directory.exists():
                for path in directory.rglob('*'):
                    if path.is_symlink():
                        raise ValueError(f'Symlink in weight path: {path}')
                    relative = path.relative_to(root)
                    if path.is_file() and str(relative) not in records:
                        if '.rsync-partial' not in relative.parts:
                            raise ValueError(f'Unexpected model file: {relative}')
    total = sum(record['bytes'] for record in records.values())
    completed = 0
    started = time.monotonic()
    last_report = started
    for number, (name, record) in enumerate(records.items(), 1):
        if not name or name.startswith('/') or any(x in ('', '.', '..') for x in name.split('/')) or '\\' in name:
            raise ValueError(f'Unsafe weight path: {name}')
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f'Symlink in weight path: {path}')
        if preflight:
            if path.exists() and not path.is_file():
                raise ValueError(f'Destination is not a regular file: {path}')
            continue
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError(f'Weight size mismatch: {name}')
        if not check_hashes:
            continue
        print(f'SHA-256 [{number}/{len(records)}] {name} ({record["bytes"] / 1024**3:.2f} GiB)', flush=True)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
                completed += len(block)
                now = time.monotonic()
                if now - last_report >= 5:
                    rate = completed / max(now - started, 0.001) / 1024**2
                    print(f'SHA-256 progress: {completed / 1024**3:.2f}/{total / 1024**3:.2f} GiB, {rate:.1f} MiB/s', flush=True)
                    last_report = now
        if digest.hexdigest() != record['sha256']:
            raise ValueError(f'Weight checksum mismatch: {name}')
        print(f'SHA-256 verified: {name}', flush=True)


def prepare_weights(root, *, check_hashes=True):
    root = Path(root).expanduser().absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('Weight root must not contain symlinks')
    files, stamps = scan_directory(root)
    manifests = sorted(root.glob('*/*/*/cloud_source.json'))
    if not manifests:
        raise ValueError('No repository/model/revision/cloud_source.json manifests found')
    records = {}
    for path in manifests:
        manifest = json.loads(path.read_text())
        relative = path.parent.relative_to(root)
        if '/'.join(relative.parts[:2]) != manifest['repository'] or relative.parts[2] != manifest['revision']:
            raise ValueError(f'Manifest repository/revision disagrees with directory: {path}')
        entries = manifest['files']
        if not entries or 'config.json' not in entries or any(n in entries for n in ('adapter_config.json', 'cloud_source.json')):
            raise ValueError(f'Incomplete or invalid base-weight manifest: {path}')
        for name, record in entries.items():
            if not isinstance(name, str) or not name or name.startswith('/') or any(x in ('', '.', '..') for x in name.split('/')) or '\\' in name:
                raise ValueError(f'Unsafe manifest file name: {name}')
            if not isinstance(record['bytes'], int) or record['bytes'] < 0 or not re.fullmatch('[0-9a-f]{64}', record['sha256']):
                raise ValueError(f'Invalid checksum record: {name}')
            records[str(relative / name)] = {k: record[k] for k in ('bytes', 'sha256')}
        records[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    if set(records) != {str(p.relative_to(root)) for p in files}:
        raise ValueError('Weights contain missing or unmanifested files (including partial downloads)')
    check_weight_files(root, records, check_hashes=check_hashes)
    plan = TransferPlan(root, sorted(records), stamps, {'models': len(manifests)},
                        sum(r['bytes'] for r in records.values()), records)
    plan.verify_unchanged()
    return plan


def verify_weights_destination(plan, destination, preflight=False):
    if ':' not in destination:
        check_weight_files(destination, plan.checksums, preflight)
        return
    host, root = destination.split(':', 1)
    code = source_inspect.getsource(check_weight_files) + "\nimport json,sys\np=json.load(sys.stdin)\ncheck_weight_files(**p)\n"
    command = ['ssh', host, shlex.join(['python3', '-c', code])]
    result = subprocess.run(command, input=json.dumps(dict(root=root, records=plan.checksums, preflight=preflight)), text=True)
    if result.returncode:
        raise ValueError('Remote weight path check or SHA-256 verification failed')


def validate_destination(destination, root):
    if ':' in destination:
        if not re.fullmatch(r'(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9][A-Za-z0-9_.-]*:/[^\n\r\x00]+', destination):
            raise ValueError('Use an SSH destination such as user@host:/absolute/dataset/path')
        if Path(destination.split(':', 1)[1]) == Path('/'):
            raise ValueError('Choose a dataset directory, not the remote filesystem root')
    else:
        target = Path(destination).expanduser().resolve()
        if target == Path('/') or target.is_relative_to(root) or root.is_relative_to(target):
            raise ValueError('Local destination must be separate from the source dataset')
        destination = str(target)
    return destination.rstrip('/') + '/'


def transfer(plan, destination, inspect=False):
    destination = validate_destination(destination, plan.root)
    executable = shutil.which('rsync')
    if executable is None:
        raise ValueError('rsync must be installed on the source and destination machines')
    plan.verify_unchanged()
    if plan.checksums is not None:
        verify_weights_destination(plan, destination, preflight=True)
    with tempfile.TemporaryDirectory(prefix='triage-transfer-') as temporary:
        listing = Path(temporary) / 'files.list'
        listing.write_bytes(b''.join(os.fsencode(path) + b'\0' for path in plan.files))
        command = [executable, '-rlt', '--protect-args', '--from0', f'--files-from={listing}',
                   '--partial-dir=.rsync-partial', '--stats', '--human-readable']
        if plan.checksums is not None and not inspect:
            command.append('--checksum')
        if inspect:
            command.extend(['--dry-run', '--itemize-changes'])
        else:
            command.append('--info=progress2')
        command.extend(['--', str(plan.root) + '/', destination])
        result = subprocess.run(command, check=False)
    if result.returncode == 0:
        plan.verify_unchanged()
        if plan.checksums is not None and not inspect:
            verify_weights_destination(plan, destination)
            check_weight_files(plan.root, plan.checksums)
            plan.verify_unchanged()
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument('--weights-root', type=Path, help='Verified base weights with repository/model/revision manifests')
    roots.add_argument('--media-root', type=Path, help='Parent of prepared 06, 07, and 08 directories')
    parser.add_argument('--split-dir', type=Path, help='Saved split directory, absolute or relative to media-root')
    parser.add_argument('--destination', required=True, help='user@host:/absolute/dataset/path or local directory')
    parser.add_argument('--inspect', action='store_true', help='Compare files and sizes with destination without copying')
    args = parser.parse_args(argv)
    try:
        if args.weights_root is not None:
            if args.split_dir is not None:
                parser.error('--split-dir is only used with --media-root')
            validate_destination(args.destination, args.weights_root.expanduser().resolve())
            print('Checking weight manifests, paths, and sizes (preview; no content hashing)...' if args.inspect else 'Verifying source weight SHA-256 checksums...', flush=True)
            plan = prepare_weights(args.weights_root, check_hashes=not args.inspect)
        else:
            if args.split_dir is None:
                parser.error('--split-dir is required with --media-root')
            root = args.media_root.expanduser().resolve()
            split = args.split_dir.expanduser()
            if not split.is_absolute():
                split = root / split
            validate_destination(args.destination, root)
            print('Checking prepared manifests, clip files, and split...', flush=True)
            plan = prepare_transfer(root, split)
        for variant, count in plan.counts.items():
            print(f'{variant}: {count}' if plan.checksums is not None else f'{variant}: {count} prepared clips', flush=True)
        print(f'{len(plan.files):,} files; {plan.total_bytes / 1024**3:.2f} GiB source size', flush=True)
        print('Preview: no files copied.' if args.inspect else 'Copying; existing destination files are not deleted.', flush=True)
        result = transfer(plan, args.destination, args.inspect)
        if result:
            print(f'rsync exited with code {result}; resolve the reported error and rerun the same command.', file=sys.stderr)
        else:
            print(('Preview complete; weight contents were not checksum-verified.' if plan.checksums is not None else 'Preview complete.') if args.inspect else ('Transfer complete; source and destination SHA-256 checks passed.' if plan.checksums is not None else 'Transfer complete; source metadata remained unchanged.'))
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Dataset transfer failed: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nCancelled. Rerun the same command to resume copying.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
