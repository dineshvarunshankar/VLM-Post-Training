"""Create positive-rich train/validation/test references in one dated directory."""

import argparse
from datetime import date
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
from scripts.train.shared.storage_settings import local_dataset_directory, annotation_dataset_directory
from scripts.train.shared.paths import REPO_ROOT, CONFIG_DIR
from scripts.train.splits.inventory import load_inventory, verify_inputs
from scripts.train.splits.reporting import summary_report, print_report
from scripts.train.splits.selection import select_splits, validate_percentage


def arguments(argv=None):
    load_dotenv(REPO_ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-root', type=Path, default=local_dataset_directory() or annotation_dataset_directory(), help='Parent of 06 and 07; local dataset root, otherwise AirLab dataset root')
    parser.add_argument('--split-root', type=Path, help='Defaults to media-root/08-train_val_test')
    parser.add_argument('--date', default=date.today().isoformat(), help='YYYY-MM-DD; split date; answer-only files go inside <date>/answer_only')
    parser.add_argument('--percentage', type=float, help='Percentage for EACH validation/test split; default 5')
    parser.add_argument('--modality', type=str.upper, choices=['RGB', 'IR'], help='Required without an interactive terminal; otherwise prompts')
    parser.add_argument('--catalog', type=Path, help='Defaults to the RGB catalog; IR requires an explicit modality-specific catalog')
    parser.add_argument('--completed-manifest', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=8, help='Concurrent annotation readers; no video decoding')
    parser.add_argument('--inspect', action='store_true', help='Print the proposed split without writing any files')
    args = parser.parse_args(argv)
    try:
        if args.percentage is None:
            args.percentage = float((input('Percentage for each validation/test split [5]: ').strip() or '5') if sys.stdin.isatty() else 5)
        if args.modality is None:
            if not sys.stdin.isatty():
                parser.error('Provide --modality RGB or --modality IR')
            args.modality = (input('Modality [RGB/IR, default RGB]: ').strip() or 'RGB').upper()
        if args.modality not in {'RGB', 'IR'}:
            parser.error('Modality must be RGB or IR')
        validate_percentage(args.percentage)
        if date.fromisoformat(args.date).isoformat() != args.date:
            parser.error('Date must use YYYY-MM-DD')
        if args.workers < 1:
            parser.error('workers must be positive')
        if args.media_root is None:
            parser.error('Provide --media-root or set TRIAGE_DATASET_ROOT')
        args.media_root = args.media_root.expanduser().resolve()
        args.split_root = (args.split_root or args.media_root / '08-train_val_test').expanduser().resolve()
        for folder in ('04-segmented_snippets', '05-segmented_stabilized_snippets', '06-native_media_dataset', '07-stabilized_media_dataset'):
            if args.split_root.is_relative_to(args.media_root / folder):
                parser.error('Split output must be outside 04, 05, 06, and 07')
        if args.catalog is None:
            if args.modality != 'RGB':
                parser.error('IR requires --catalog with thermal-specific definitions; RGB definitions are not reused')
            args.catalog = CONFIG_DIR / 'catalogs/RGB/label_catalog.yaml'
        args.catalog = args.catalog.expanduser().resolve()
        if args.modality == 'IR' and args.catalog == (CONFIG_DIR / 'catalogs/RGB/label_catalog.yaml').resolve():
            parser.error('The RGB catalog cannot be used for IR splits')
        destination = args.split_root / args.date / "answer_only"
        if not args.inspect and (destination.exists() or destination.is_symlink()):
            parser.error(f'Answer-only split already exists; choose another date: {destination}')
    except (ValueError, EOFError) as error:
        parser.error(str(error))
    return args


def publish_split(destination, selection, report):
    """Publish an answer-only split; never replace an existing split."""
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'Answer-only split already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / ('.' + destination.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with tempfile.TemporaryDirectory(prefix='.' + destination.name + '-', dir=destination.parent) as temporary:
            staging = Path(temporary) / 'split'
            staging.mkdir()
            split_hashes = {}
            for name in ('train', 'validation', 'test'):
                rows = [row for clip in selection[name] for row in clip.rows]
                rows.sort(key=lambda row: (row['variant'], row['clip'], row['category']))
                content = ''.join(json.dumps(row) + '\n' for row in rows)
                (staging / (name + '.jsonl')).write_text(content)
                split_hashes[name] = hashlib.sha256(content.encode()).hexdigest()
            (staging / 'summary.json').write_text(json.dumps({**report, 'split_sha256': split_hashes}, indent=2, allow_nan=False) + '\n')
            if destination.exists() or destination.is_symlink():
                raise ValueError(f'Answer-only split already exists: {destination}')
            staging.rename(destination)
    finally:
        lock.unlink()


def main(argv=None):
    args = arguments(argv)
    try:
        print('Reading completed prepared annotations...', flush=True)
        candidates, metadata, inputs = load_inventory(args.media_root, args.catalog, args.modality,
                                                      args.completed_manifest, args.workers)
        print(f'Selecting from {len(candidates)} clip pairs...', flush=True)
        selection = select_splits(candidates, args.percentage, args.seed)
        destination = args.split_root / args.date / "answer_only"
        report = summary_report(selection, candidates, metadata, inputs, args.media_root,
                                args.date, args.percentage, args.seed)
        if not args.inspect:
            verify_inputs(inputs, Path(metadata['completion_manifest']['path']),
                          metadata['completion_manifest']['sha256'], args.workers)
            publish_split(destination, selection, report)
        print_report(report, destination, args.inspect)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Split preparation failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
