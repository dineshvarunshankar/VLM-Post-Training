"""Read-only dataset inventory without video decoding or video-content hashing."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def file_stamp(path):
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def annotation_stamps(folder):
    files = sorted(folder.rglob('*'))
    if folder.is_symlink() or any(path.is_symlink() for path in files):
        raise ValueError('Annotation symlinks are not supported')
    return {str(path.relative_to(folder)): file_stamp(path) for path in files if path.is_file()}


# These writers use the same overlay, encoding, and verification implementation.
# 959da46: only eligibility and publication membership checks changed.
COMPATIBLE_FLAT_WRITERS = {'7062994fa1d497cb62e170d507ab618c52b399eb3ce3b74119cd28e26ed41719', 'bc4515396854c7ade38142bc3ab2e1a689de56be8bd7f853844129678fff4cc9', '959da46ec98c2a8e89a479a70878ab02a0e8e0999ee5ec5585405ef1386db016', '3e15a9e3e44ab069825691c8c000de9d7bf9dc3d18695792956681df68fd2621', '9cd4792a7446122995e84b6983b1f1e4ed233769ee2d2e4b02288fbc2680850b', '4d0f8e93a683ece1ce9dacafb115d10d27ba08fda0d43d9cf2b082099418ed32'}


def output_paths(video, clip_name=None):
    return {'video': video, 'sidecar': video.with_suffix('.json'),
            'annotations': video.parent / 'all_annotations',
            'provenance': video.parent / 'provenance.json'}


def flat_output_paths(video, clip_name):
    dataset = video.parent.parent
    return {'video': dataset / video.name, 'sidecar': dataset / (video.stem + '.json'),
            'annotations': dataset / '00-medical_annotations' / clip_name,
            'provenance': dataset / '00-provenance' / (video.stem + '.json')}



def snapshot(video, annotations, output_video, output_annotations, output_sidecar):
    return {'source_video': file_stamp(video),
            'source_sidecar': file_stamp(video.with_suffix('.json')),
            'source_annotations': annotation_stamps(annotations),
            'output_video': file_stamp(output_video),
            'output_sidecar': file_stamp(output_sidecar),
            'output_annotations': annotation_stamps(output_annotations)}


def output_status(video, annotations, output, writer_hash, box_px):
    paths = output_paths(output, annotations.name)
    if not output.is_file():
        if flat_output_paths(output, annotations.name)['video'].is_file():
            return 'needs_verification', 'Existing flat output will be checked and moved into the clip folder'
        if any(p.exists() for p in paths.values()):
            return 'needs_verification', 'Partial output exists without its MP4'
        return 'missing', 'No generated output'
    try:
        saved = json.loads(paths['provenance'].read_text())
        signature = saved.get('source_signature', {})
        if (saved.get('verification') != 'decoded frame hashes match uncompressed filtered source' or
                signature.get('source_video') != str(video.resolve()) or
                signature.get('source_annotations') != str(annotations.resolve())):
            return 'needs_verification', 'Saved verification or source identity does not match'
        if (signature.get('writer_sha256') not in ({writer_hash} | COMPATIBLE_FLAT_WRITERS) or
                signature.get('box_px') != box_px or signature.get('box_color') != '0x00ff00'):
            return 'changed', 'Writer code or overlay settings changed'
        previous = saved.get('inspection_snapshot')
        if not previous:
            return 'needs_verification', 'Older output has no fast-inspection snapshot'
        current = snapshot(video, annotations, output, paths['annotations'], paths['sidecar'])
        if current != previous:
            return 'changed', 'File size, modification time, or annotation file list changed'
        return 'current_by_metadata', 'File sizes and modification times match; contents were not rehashed'
    except (OSError, ValueError, TypeError, AttributeError):
        return 'needs_verification', 'Output provenance or source files need validation'


def json_object(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def inspect_inventory(source_root, output_root, variants, rejected, accepted, selection, box_px, writer, limit=None, completion_manifest=None):
    if completion_manifest is not None:
        return inspect_completed_inventory(source_root, output_root, variants, box_px, writer, limit, completion_manifest)
    writer_hash = hashlib.sha256(writer.read_bytes()).hexdigest()
    records, orphan_outputs, unmatched_annotations = [], [], []
    source_counts = {}
    for variant, (source_name, output_name) in variants.items():
        source = source_root / source_name
        output_base = output_root / output_name
        videos = {p.name: p for p in source.glob('*.mp4') if p.is_file()}
        source_counts[variant] = len(videos)
        names = json_object(source / '_names.json')
        annotation_root = source / '00-medical_annotations'
        folders = {p.name: p for p in annotation_root.iterdir() if p.is_dir()}
        metadata = {name: json_object(folder / 'meta.json') for name, folder in folders.items()}
        by_video = defaultdict(set)
        for name, meta in metadata.items():
            value = meta.get('video_path')
            if isinstance(value, str) and value:
                by_video[Path(value).name].add(name)
        associated = set()
        inventory = []
        for filename, video in videos.items():
            candidates = set(by_video.get(filename, ()))
            entry = names.get(video.stem, {})
            entry = entry if isinstance(entry, dict) else {}
            full_name = entry.get('full_filename')
            formal = Path(full_name).stem if isinstance(full_name, str) and full_name else None
            sidecar = {}
            if not formal and not candidates:
                sidecar = json_object(video.with_suffix('.json'))
                full_name = sidecar.get('full_video_filename')
                formal = Path(full_name).stem if isinstance(full_name, str) and full_name else sidecar.get('clip')
                if not isinstance(formal, str):
                    formal = None
            if formal in folders:
                annotated_video = metadata[formal].get('video_path')
                if not annotated_video or (isinstance(annotated_video, str) and Path(annotated_video).name == filename):
                    candidates.add(formal)
            associated.update(candidates)
            clip_name = next(iter(candidates)) if len(candidates) == 1 else (formal or video.stem)
            inventory.append((clip_name, video, candidates, sidecar))
        selected_count = 0
        for clip_name, video, candidates, sidecar in sorted(inventory, key=lambda item: (item[0], item[1].name)):
            output = output_base / video.stem / video.name
            is_rejected = (clip_name.removesuffix('_S') in rejected or
                           any(name.removesuffix('_S') in rejected for name in candidates))
            record = {'variant': variant, 'clip': clip_name, 'source_video': str(video),
                      'modality': sidecar.get('modality') or 'unknown',
                      'completion': 'rejected' if is_rejected else 'unknown', 'rejected': is_rejected,
                      'created': output.is_file() or (output_base / video.name).is_file(), 'selected': False, 'eligible': False,
                      'destination': str(output), 'status': 'needs_verification'}
            records.append(record)
            if len(candidates) != 1:
                record['reason'] = ('Multiple annotation folders map to this video' if candidates
                                    else 'No annotation folder mapped to this video')
                continue
            folder = folders[clip_name]
            meta = metadata[clip_name]
            record['source_annotations'] = str(folder)
            if not meta:
                record['reason'] = 'Missing or invalid meta.json'
                continue
            record['modality'] = meta.get('modality') or record['modality']
            status = meta.get('annotation_status')
            record['annotation_status'] = status
            if not is_rejected and isinstance(status, str) and status:
                record['completion'] = {'completed_minimal': 'minimal', 'completed': 'full'}.get(
                    status, 'not_completed_not_rejected')
            value = meta.get('video_path')
            if not isinstance(value, str) or Path(value).name != video.name:
                record['reason'] = 'Annotation video_path does not match this source MP4'
                continue
            record['eligible'] = not is_rejected and (selection == 'all-non-rejected' or (isinstance(status, str) and status in accepted))
            if record['eligible'] and (limit is None or selected_count < limit):
                record['selected'] = True
                selected_count += 1
            record['status'], record['reason'] = output_status(video, folder, output, writer_hash, box_px)
            required = ['meta.json', 'boxes.json']
            if selection == 'completed':
                required.append('labels.json')
            if not video.with_suffix('.json').is_file() or any(not (folder / name).is_file() for name in required):
                record.update(status='needs_verification', reason='Required source file is missing')
        for name in sorted(set(folders) - associated):
            unmatched_annotations.append({'variant': variant, 'clip': name, 'source_annotations': str(folders[name]),
                'video_path': metadata[name].get('video_path'),
                'reason': 'Annotation folder has no matching source MP4'})
        if output_base.is_dir():
            for video in sorted([*output_base.glob('*.mp4'), *output_base.glob('*/*.mp4')]):
                if video.name not in videos:
                    orphan_outputs.append({'variant': variant, 'destination': str(video)})
    totals = Counter()
    grouped = {}
    actions = {variant: Counter() for variant in variants}
    eligibility = {variant: Counter() for variant in variants}
    for record in records:
        eligibility.setdefault(record['variant'], Counter())[record['completion']] += 1
        action = 'excluded'
        if record['selected']:
            action = {'missing': 'create', 'changed': 'update', 'current_by_metadata': 'skip'}.get(record['status'], 'verify')
        record['expected_action'] = action
        actions.setdefault(record['variant'], Counter())[action] += 1
        key = (record['variant'], record['modality'], record['completion'])
        group = grouped.setdefault(key, Counter())
        for counter in (totals, group):
            counter['total'] += 1
            counter['created'] += int(record['created'])
            counter['selected'] += int(record['selected'])
            counter['rejected'] += int(record['rejected'])
            counter[record['status']] += 1
    columns = ['total', 'created', 'missing', 'changed', 'current_by_metadata', 'needs_verification', 'rejected', 'selected']
    rows = [{'variant': key[0], 'modality': key[1], 'completion': key[2],
             **{name: count[name] for name in columns}} for key, count in sorted(grouped.items())]
    return {'schema_version': 2, 'mode': 'inspect', 'selection': selection, 'limit': limit,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_root': str(source_root), 'output_root': str(output_root),
            'summary': {name: totals[name] for name in columns}, 'breakdown': rows,
            'completion_counts': {variant: {key: counts[key] for key in
                ('minimal', 'full', 'not_completed_not_rejected', 'rejected', 'unknown')}
                for variant, counts in eligibility.items()},
            'expected_actions': {variant: {key: counts[key] for key in ('create', 'update', 'skip', 'verify', 'excluded')}
                                 for variant, counts in actions.items()},
            'source_video_counts': source_counts, 'unmatched_annotations': unmatched_annotations,
            'unmatched_created_outputs': orphan_outputs, 'clips': records}


def inspect_completed_inventory(source_root, output_root, variants, box_px, writer, limit, manifest):
    writer_hash = hashlib.sha256(writer.read_bytes()).hexdigest()
    records, source_counts, actions, completion_counts, unmatched = [], {}, {}, {}, []
    for variant, (source_name, output_name) in variants.items():
        source, output = source_root / source_name, output_root / output_name
        videos = {p.name: p for p in source.glob('*.mp4')}
        source_counts[variant] = len(videos)
        entries = sorted(manifest.entries(variant), key=lambda entry: entry['clip'])
        counts = Counter(entry['level'] for entry in entries)
        listed = {entry['video'].name for entry in entries}
        counts['not_in_completed_manifest'] = len(set(videos) - listed)
        completion_counts[variant] = dict(counts)
        generated = {p.name for p in output.glob('*/*.mp4')} | {p.name for p in output.glob('*.mp4')}
        for index, entry in enumerate(entries):
            video, folder = entry['video'], entry['annotations']
            destination = output / video.stem / video.name
            meta = json_object(folder / 'meta.json')
            selected = limit is None or index < limit
            record = {'variant': variant, 'clip': entry['clip'], 'source_video': str(video),
                'source_annotations': str(folder), 'destination': str(destination),
                'modality': meta.get('modality') or 'unknown', 'completion': entry['level'],
                'rejected': False, 'created': video.name in generated, 'selected': selected,
                'eligible': True, 'status': 'needs_verification'}
            required = ['meta.json', 'boxes.json', 'labels.json']
            if video.name not in videos:
                record['reason'] = 'Completed manifest video is missing'
            elif Path(meta.get('video_path') or '').name != video.name:
                record['reason'] = 'Export video_path differs from current completion manifest'
            elif not video.with_suffix('.json').is_file() or any(not (folder / name).is_file() for name in required):
                record['reason'] = 'Required source file is missing for a completed clip'
            elif selected:
                record['status'], record['reason'] = output_status(video, folder, destination, writer_hash, box_px)
            else:
                record['status'] = 'not_inspected'
                record['reason'] = 'Outside --limit; output freshness not inspected'
            record['expected_action'] = ({'missing': 'create', 'changed': 'update', 'current_by_metadata': 'skip'}
                .get(record['status'], 'verify') if selected else 'excluded')
            records.append(record)
        for filename in sorted(set(videos) - listed):
            records.append({'variant': variant, 'clip': videos[filename].stem,
                'source_video': str(videos[filename]), 'modality': 'unknown',
                'completion': 'not_in_completed_manifest', 'rejected': False,
                'created': filename in generated, 'selected': False, 'eligible': False,
                'status': 'not_in_completed_manifest', 'expected_action': 'excluded',
                'reason': 'Absent from current site completion list; exported status is not used'})
        actions[variant] = dict(Counter(r['expected_action'] for r in records if r['variant'] == variant))
        for filename in sorted(generated - listed):
            unmatched.append({'variant': variant, 'video': filename,
                'reason': 'Existing output is not in current completion manifest; left untouched'})
    columns = ['total', 'created', 'missing', 'changed', 'current_by_metadata', 'needs_verification', 'rejected', 'selected']
    groups, totals = {}, Counter()
    for record in records:
        group = groups.setdefault((record['variant'], record['modality'], record['completion']), Counter())
        for counter in (group, totals):
            counter['total'] += 1
            counter['created'] += int(record['created'])
            counter['selected'] += int(record['selected'])
            counter[record['status']] += 1
    return {'schema_version': 3, 'mode': 'inspect', 'selection': 'completed', 'limit': limit,
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(source_root),
        'output_root': str(output_root), 'completion_manifest': manifest.summary(),
        'source_video_counts': source_counts, 'completion_counts': completion_counts,
        'expected_actions': {variant: {key: counts.get(key, 0) for key in ('create', 'update', 'skip', 'verify', 'excluded')}
                             for variant, counts in actions.items()},
        'summary': {key: totals[key] for key in columns},
        'breakdown': [{'variant': key[0], 'modality': key[1], 'completion': key[2],
                       **{column: counts[column] for column in columns}} for key, counts in sorted(groups.items())],
        'unmatched_annotations': [], 'unmatched_created_outputs': unmatched, 'clips': records}


def print_summary(report):
    print('Source MP4 inventory (one record per video):')
    for variant, count in report['source_video_counts'].items():
        print(f'{variant}: {count}')
    print('\nAnnotation status (separate, non-overlapping counts):')
    for variant, counts in report['completion_counts'].items():
        if 'completion_manifest' in report:
            print(f"{variant}: minimal={counts.get('minimal', 0)}, full={counts.get('full', 0)}, "
                  f"source MP4s outside completion list={counts.get('not_in_completed_manifest', 0)}")
            continue
        print(f"{variant}: minimal={counts['minimal']}, full={counts['full']}, "
              f"not completed and not rejected={counts['not_completed_not_rejected']}, "
              f"rejected={counts['rejected']}, unknown/missing status={counts['unknown']}")
    print('\nExpected next run (default create/update mode):')
    print(f"{'Variant':12} {'Create':>8} {'Update':>8} {'Skip*':>8} {'Verify':>8} {'Excluded':>10}")
    for variant, counts in report['expected_actions'].items():
        print(f"{variant:12} {counts['create']:8} {counts['update']:8} {counts['skip']:8} {counts['verify']:8} {counts['excluded']:10}")
    print('\nInventory by modality and completion:')
    columns = [('variant', 'Variant'), ('modality', 'Modality'), ('completion', 'Completion'),
               ('total', 'Total'), ('created', 'Created'), ('missing', 'Missing'),
               ('changed', 'Changed'), ('current_by_metadata', 'Current*'),
               ('needs_verification', 'Verify'), ('rejected', 'Rejected'), ('selected', 'Selected')]
    rows = report['breakdown'] + [{'variant': 'TOTAL', 'modality': '', 'completion': '', **report['summary']}]
    widths = [max(len(label), *(len(str(row[key])) for row in rows)) for key, label in columns]
    print('  '.join(label.ljust(width) for (_, label), width in zip(columns, widths)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        print('  '.join(str(row[key]).ljust(width) for (key, _), width in zip(columns, widths)))
    print('\nCreated = output MP4 exists; rejected clips can still have older outputs.')
    print('Changed = saved file metadata or writer settings differ; contents are checked during generation.')
    print('* Current = sizes and modification times match; video contents were not rehashed.')
    print('Verify = missing metadata, an older output without a snapshot, or incomplete source/output files.')
    print('Selected = clips included by completion/rejection rules and --limit; not a media-validity guarantee.')
    if 'completion_manifest' in report:
        print('Excluded = source MP4 absent from the current completion list, or outside --limit.')
        print('The completed-only list cannot distinguish reopened, in-progress, skipped, and rejected clips.')
    else:
        print('Excluded includes videos without usable annotation metadata; their completion status is unknown.')
    if report['unmatched_annotations']:
        print(f"Annotation folders without a matching source MP4 (not included in video totals): {len(report['unmatched_annotations'])}")
    if report['unmatched_created_outputs']:
        print(f"Unmatched existing output MP4s: {len(report['unmatched_created_outputs'])}")
    print('No videos decoded or hashed. Use --report PATH to save per-clip details as JSON.')
