"""Read completed prepared clip pairs and verify the annotations used for selection."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

from scripts.preprocess.completion_manifest import CompletionManifest, DEFAULT_MANIFEST
from scripts.train.shared.data import DATASETS, extract_answers, read_yaml, recording_group, sha256
from scripts.train.splits.selection import ClipPair


def label_sets(answers, catalog):
    observed, absent, unassessable = set(), set(), set()
    for category, answer in answers.items():
        if category in {'context', 'subject_check'}:
            continue
        definition = catalog['categories'][category]
        for section, group in definition['subcategories'].items():
            for cue in group['cues']:
                name = cue['name']
                if answer[name] != 1:
                    continue
                key = f'{category}/{name}'
                if section == 'observables':
                    observed.add(key)
                elif section == 'assessment':
                    if name.endswith(('_assessed_absent', '_assessed_no_injury')):
                        absent.add(key)
                    elif name.endswith('_not_assessable') or '_view_inadequate' in name:
                        unassessable.add(key)
    return observed, absent, unassessable


def load_inventory(root, catalog_path, modality, manifest_path=None, workers=8):
    root, catalog_path = Path(root).resolve(), Path(catalog_path).resolve()
    catalog_hash = sha256(catalog_path)
    catalog = read_yaml(catalog_path)
    if sha256(catalog_path) != catalog_hash:
        raise ValueError('Catalog changed during loading')
    manifest = CompletionManifest(manifest_path or root / DEFAULT_MANIFEST, root)
    snapshot = manifest.summary()
    native = manifest.entries('native')
    stabilized = {entry['clip'].removesuffix('_S'): entry for entry in manifest.entries('stabilized')}
    inputs = {catalog_path: catalog_hash}

    def read_pair(entry):
        fingerprints = {}

        def read(path):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Input escapes the media root or is a symlink: {path}')
            content = path.read_bytes()
            fingerprints[path] = hashlib.sha256(content).hexdigest()
            return json.loads(content)

        native_dir = root / DATASETS['native'] / entry['clip_id']
        meta = read(native_dir / 'all_annotations/meta.json')
        if meta.get('modality') != modality:
            return None, fingerprints
        rows, observed, absent, unassessable = [], set(), set(), set()
        details = {'variants': {}, 'excluded_categories': {}}
        identity = None
        entries = {'native': entry, 'stabilized': stabilized[entry['clip']]}
        for variant, current in entries.items():
            folder = root / DATASETS[variant] / current['clip_id']
            annotations = folder / 'all_annotations'
            metadata = meta if variant == 'native' else read(annotations / 'meta.json')
            if metadata.get('clip') != current['clip'] or metadata.get('modality') != modality:
                raise ValueError(f'{variant}/{current["clip_id"]}: completion identity or modality mismatch')
            video = folder / (current['clip_id'] + '.mp4')
            if not video.is_file() or video.is_symlink() or Path(metadata.get('video_path', '')).name != video.name:
                raise ValueError(f'{variant}/{current["clip_id"]}: missing or mismatched prepared video')
            provenance = read(folder / 'provenance.json')
            labels = read(annotations / 'labels.json')
            camera = read(annotations / 'camera_motion.json')
            if not isinstance(camera.get('view_type'), str) or not camera['view_type'].strip():
                raise ValueError(f'{variant}/{current["clip_id"]}: missing camera viewpoint')
            sidecar_path = video.with_suffix('.json')
            sidecar = read(sidecar_path)
            for name in ('meta.json', 'labels.json', 'camera_motion.json'):
                if fingerprints[annotations / name] != provenance.get('annotation_sha256', {}).get(name):
                    raise ValueError(f'{variant}/{current["clip_id"]}: {name} differs from writer provenance')
            if fingerprints[sidecar_path] != provenance.get('sidecar_sha256'):
                raise ValueError(f'{variant}/{current["clip_id"]}: sidecar differs from writer provenance')
            if variant == 'native':
                source, person = sidecar.get('video_name'), sidecar.get('person_id')
                if not isinstance(source, str) or not source:
                    raise ValueError(f'{variant}/{current["clip_id"]}: missing source video_name')
                full_name = sidecar.get('full_video_filename')
                if ((full_name and Path(full_name).stem != metadata['clip']) or
                        (not full_name and source != recording_group(metadata))):
                    raise ValueError(f'{variant}/{current["clip_id"]}: source clip identity mismatch')
                details['person_scope'] = source
                original_source = sidecar.get('clip_provenance', {}).get('snippet_provenance', {}).get('source_video')
                if original_source:
                    source = Path(original_source).stem
                if isinstance(person, bool) or not isinstance(person, (str, int)) or not re.fullmatch(r'\d+', str(person)):
                    raise ValueError(f'{variant}/{current["clip_id"]}: missing or invalid person_id')
                year = re.match(r'^(Y\d+)_', source)
                if not year:
                    raise ValueError(f'{variant}/{current["clip_id"]}: source has no dataset-year prefix')
                identity = (source, str(person), year.group(1))
            elif (sidecar.get('source_clip') != entry['clip'] or
                  sidecar.get('source_video_filename') != entry['clip_id'] + '.mp4' or
                  recording_group(metadata) != recording_group(meta)):
                raise ValueError(f'{entry["clip_id"]}: stabilized sidecar does not link to the paired native clip')
            answers, excluded = extract_answers(labels, catalog, variant)
            if not answers:
                raise ValueError(f'{variant}/{current["clip_id"]}: no valid category answers for this catalog')
            details['excluded_categories'][variant] = excluded
            for category in sorted(answers):
                rows.append({'variant': variant, 'clip': current['clip_id'], 'category': category})
            obs, absents, unknowns = label_sets(answers, catalog)
            observed.update(obs)
            absent.update(absents)
            unassessable.update(unknowns)
            expected_gt = {'categories': answers, 'source_hashes': {
                'labels_sha256': fingerprints[annotations / 'labels.json'], 'label_catalog_sha256': catalog_hash}}
            gt_path = folder / 'GT/medical_vqa.json'
            if not gt_path.exists():
                gt_status = 'missing'
            else:
                try:
                    gt_status = 'current' if read(gt_path) == expected_gt else 'stale'
                except (ValueError, OSError):
                    gt_status = 'invalid'
            details['variants'][variant] = {
                'clip': current['clip_id'], 'categories': len(answers), 'gt_status': gt_status,
                'retained_rejection_marker': (annotations / 'rejected.json').exists(),
                'positive_labels': sorted(f'{category}/{key}' for category, answer in answers.items()
                                          for key, value in answer.items() if value == 1),
                'video_sha256': provenance.get('output_sha256')}
        return ClipPair(entry['clip_id'], *identity, tuple(rows), frozenset(observed),
                        frozenset(absent), frozenset(unassessable), details), fingerprints

    candidates, errors = [], []
    def checked(entry):
        try:
            return read_pair(entry), None
        except (OSError, ValueError, TypeError, KeyError) as error:
            return None, {'clip': entry['clip_id'], 'reason': str(error)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result, error in pool.map(checked, native):
            if error:
                errors.append(error)
                continue
            pair, fingerprints = result
            inputs.update(fingerprints)
            if pair is not None:
                candidates.append(pair)
    if errors:
        raise ValueError('Split input validation failed:\n' + json.dumps(errors, indent=2))
    if manifest.summary()['sha256'] != snapshot['sha256']:
        raise ValueError('Completion list changed during inventory; retry')
    if not candidates:
        raise ValueError(f'No completed prepared {modality} pairs are available')
    return candidates, {'completion_manifest': snapshot, 'catalog_sha256': catalog_hash,
                        'catalog_path': str(catalog_path), 'modality': modality}, inputs


def verify_inputs(inputs, manifest_path, expected_manifest_hash, workers=8):
    def unchanged(item):
        path, expected = item
        if sha256(path) != expected:
            raise ValueError(f'Input changed during split selection: {path}')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(unchanged, inputs.items()))
    if sha256(manifest_path) != expected_manifest_hash:
        raise ValueError('Completion list changed during split selection; retry')
