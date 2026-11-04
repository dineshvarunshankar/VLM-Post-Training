"""Read-only subject-check snapshot preflight for dataset transfer."""

import json
from pathlib import Path

from scripts.train.storage.cloud_models import local_file


def subject_transfer_files(root, split, *, include_existing_subject_labels=True):
    """Return only verified media, targets, provenance and frozen split files."""
    root, split = Path(root), Path(split)
    manifest_path = local_file(root, '10-other_datasets/people_negatives_VK_20261102/dataset_manifest.json')
    if not manifest_path.is_file():
        raise ValueError(f'Missing subject-check manifest: {manifest_path}')
    from scripts.train.shared.subject_check import SUBJECT_KEYS, verified_entries
    dataset_root, manifest_path, entries = verified_entries(root)
    files = {manifest_path}
    for entry in entries:
        image = local_file(dataset_root, entry['image'])
        gt = local_file(dataset_root, str(Path(entry['image']).parent / 'GT/subject_check.json'))
        expected = {'answer': dict(zip(SUBJECT_KEYS, (0, 1))),
                    'source_hashes': {key: entry[key] for key in ('image_sha256', 'provenance_sha256')}}
        if not gt.is_file() or json.loads(gt.read_text()) != expected:
            raise ValueError(f'Missing or stale subject-check GT: {gt}')
        files.update((image, local_file(dataset_root, entry['provenance']), gt))
    sidecar = split / 'subject_check'
    summary_path = local_file(sidecar, 'summary.json')
    if not summary_path.is_file():
        raise ValueError('Missing subject-check split summary')
    summary = json.loads(summary_path.read_text())
    if summary.get('schema_version') != 1 or summary.get('kind') != 'subject_check_splits':
        raise ValueError('Invalid subject-check split summary')

    from scripts.train.shared.subject_check import load_subject_check_splits
    medical_splits = {name: [{'reference': json.loads(line)}
                              for line in (split / (name + '.jsonl')).read_text().splitlines() if line.strip()]
                      for name in ('train', 'validation', 'test')}
    rows, _ = load_subject_check_splits(root, split, {
        'enabled': True, 'modality': summary.get('modality'),
        'include_existing_subject_labels': include_existing_subject_labels,
    }, medical_splits, None, None)
    from scripts.train.shared.data import clip_path
    for split_rows in rows.values():
        for row in split_rows:
            if row['media_type'] == 'video':
                ref = row['reference']
                files.add(clip_path(root, ref['variant'], ref['clip']) / 'GT/subject_check.json')
    for name in ('train', 'validation', 'test'):
        files.add(local_file(split, name + '.jsonl'))
        files.add(local_file(sidecar, name + '.jsonl'))
    files.add(summary_path)
    for path in files:
        local_file(root, str(path.relative_to(root)))
        if not path.is_file():
            raise ValueError(f'Missing subject-check file: {path}')
    return sorted(files)
