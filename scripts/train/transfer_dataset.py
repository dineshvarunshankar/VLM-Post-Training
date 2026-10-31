#!/usr/bin/env python3
"""Preview or copy prepared datasets and a saved split using rsync."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
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
    with tempfile.TemporaryDirectory(prefix='triage-transfer-') as temporary:
        listing = Path(temporary) / 'files.list'
        listing.write_bytes(b''.join(os.fsencode(path) + b'\0' for path in plan.files))
        command = [executable, '-rlt', '--protect-args', '--from0', f'--files-from={listing}',
                   '--partial-dir=.rsync-partial', '--stats', '--human-readable']
        if inspect:
            command.extend(['--dry-run', '--itemize-changes'])
        else:
            command.append('--info=progress2')
        command.extend(['--', str(plan.root) + '/', destination])
        result = subprocess.run(command, check=False)
    if result.returncode == 0:
        plan.verify_unchanged()
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-root', type=Path, required=True, help='Parent of prepared 06, 07, and 08 directories')
    parser.add_argument('--split-dir', type=Path, required=True, help='Saved split directory, absolute or relative to media-root')
    parser.add_argument('--destination', required=True, help='user@host:/absolute/dataset/path or local directory')
    parser.add_argument('--inspect', action='store_true', help='Compare files and sizes with destination without copying')
    args = parser.parse_args(argv)
    try:
        root = args.media_root.expanduser().resolve()
        split = args.split_dir.expanduser()
        if not split.is_absolute():
            split = root / split
        validate_destination(args.destination, root)
        print('Checking prepared manifests, clip files, and split...', flush=True)
        plan = prepare_transfer(root, split)
        for variant, count in plan.counts.items():
            print(f'{variant}: {count} prepared clips', flush=True)
        print(f'{len(plan.files):,} files; {plan.total_bytes / 1024**3:.2f} GiB source size', flush=True)
        print('Preview: no files copied.' if args.inspect else 'Copying; existing destination files are not deleted.', flush=True)
        result = transfer(plan, args.destination, args.inspect)
        if result:
            print(f'rsync exited with code {result}; resolve the reported error and rerun the same command.', file=sys.stderr)
        else:
            print('Preview complete.' if args.inspect else 'Transfer complete; source metadata remained unchanged.')
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Dataset transfer failed: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nCancelled. Rerun the same command to resume copying.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
