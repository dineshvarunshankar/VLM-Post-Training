#!/usr/bin/env python3
"""Create or update lossless boxed media datasets; use --inspect for status only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from completion_manifest import CompletionManifest, DEFAULT_MANIFEST
from prepared_manifest import publish_manifests

from inspection import inspect_inventory, print_summary, snapshot, output_paths, flat_output_paths, COMPATIBLE_FLAT_WRITERS, output_status

DEFAULT_ROOT = Path('/mnt/nas/chiron/datasets/full_dataset_inventory_(DO_NOT_MODIFY!!!!)')
VARIANTS = {
    'native': ('04-segmented_snippets', '06-native_media_dataset'),
    'stabilized': ('05-segmented_stabilized_snippets', '07-stabilized_media_dataset'),
}
ACCEPTED = {'completed_minimal', 'completed'}
# triage_mapping/clips.py: ClipConfig.box_px=2; draw_box color=(0, 255, 0).
COLOR = '0x00ff00'
MEDIA_THREADS = 0


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def run(command):
    command = list(command)
    if MEDIA_THREADS and command[0] == 'ffprobe':
        if '-threads' in command:
            command[command.index('-threads') + 1] = str(MEDIA_THREADS)
        else:
            command[1:1] = ['-threads', str(MEDIA_THREADS)]
    elif MEDIA_THREADS and command[0] == 'ffmpeg':
        command[1:1] = ['-filter_threads', str(MEDIA_THREADS)]
        input_index = command.index('-i')
        command[input_index:input_index] = ['-threads', str(MEDIA_THREADS)]
        command[-1:-1] = ['-threads', str(MEDIA_THREADS)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip()[-3000:] or 'Command failed')
    return result.stdout


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def pair_id(name):
    return name.removesuffix('_S')


def rejected_ids(source_root):
    """A rejection file takes precedence over any stale completion metadata."""
    rejected = set()
    for source_name, _ in VARIANTS.values():
        directory = source_root / source_name / '00-medical_annotations'
        for folder in sorted(directory.iterdir()):
            if folder.is_dir() and (folder / 'rejected.json').exists():
                rejected.add(pair_id(folder.name))
    return rejected


def probe(path):
    data = json.loads(run([
        'ffprobe', '-threads', '0', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,pix_fmt,avg_frame_rate,nb_read_frames,color_range,color_space,color_transfer,color_primaries',
        '-of', 'json', str(path),
    ]))
    stream = data['streams'][0]
    stream['frames'] = int(stream['nb_read_frames'])
    stream['fps'] = float(Fraction(stream['avg_frame_rate']))
    if stream['frames'] <= 0 or stream['fps'] <= 0:
        raise ValueError('Video has no usable frames or frame rate')
    return stream


def inspect_clip(folder, source, thickness, selection='completed', current_completion=False):
    names = ['meta.json', 'boxes.json']
    if selection == 'completed' or (folder / 'labels.json').exists():
        names.append('labels.json')
    records = {name: read_json(folder / name) for name in names}
    meta = records['meta.json']
    if selection == 'completed' and not current_completion and meta.get('annotation_status') not in ACCEPTED:
        raise ValueError('Annotation status is not eligible')
    video_name = Path(meta['video_path']).name
    video = source / video_name
    if not video.is_file() or video.is_symlink():
        raise ValueError(f'Missing video or unsupported symlink: {video_name}')
    for name, record in records.items():
        if record.get('clip') != folder.name:
            raise ValueError(f'{name}: clip identity differs from annotation folder')
        if Path(record.get('video_path', '')).name != video_name:
            raise ValueError(f'{name}: video identity differs from meta.json')
    stream = probe(video)
    width, height, count = stream['width'], stream['height'], stream['frames']
    if meta.get('resolution') != {'width': width, 'height': height}:
        raise ValueError('Annotation resolution differs from decoded video')
    if meta.get('n_frames') != count:
        raise ValueError(f'Annotation frame count {meta.get("n_frames")} differs from decoded count {count}')
    fps = float(meta['fps'])
    if not math.isfinite(fps) or not math.isclose(fps, stream['fps'], rel_tol=0.001, abs_tol=0.001):
        raise ValueError('Annotation FPS differs from video FPS')
    seen, filters = set(), []
    skipped = Counter()
    for frame in records['boxes.json']['frames']:
        index = frame['frame']
        if type(index) is not int or not 1 <= index <= count or index in seen:
            raise ValueError(f'Duplicate or invalid 1-based box frame: {index}')
        seen.add(index)
        box = frame.get('bbox_pct')
        if frame.get('enabled') is False or box is None:
            skipped['disabled_or_null'] += 1
            continue
        values = [float(box[k]) for k in ('x', 'y', 'width', 'height')]
        if not all(math.isfinite(v) for v in values) or values[2] < 0 or values[3] < 0:
            raise ValueError(f'Invalid box at frame {index}')
        if values[2] == 0 or values[3] == 0:
            skipped['zero_area'] += 1
            continue
        x, y, w, h = values
        x1, y1 = round(x * width / 100), round(y * height / 100)
        x2, y2 = round((x + w) * width / 100), round((y + h) * height / 100)
        if x2 <= x1 or y2 <= y1:
            skipped['subpixel'] += 1
            continue
        if x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
            skipped['outside_frame'] += 1
            continue
        # Retain off-screen geometry: clipping coordinates would invent a border.
        filters.append(f"drawbox=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:color={COLOR}:t={thickness}:enable='eq(n,{index-1})'")
    skipped['missing_box_record'] = count - len(seen)
    if not filters:
        raise ValueError('No visible boxes; cannot create a subject overlay')
    return video, stream, ',\n'.join(filters), dict(skipped), len(filters)


def frame_hashes(path, filter_file=None):
    command = ['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0']
    if filter_file:
        command += ['-filter_script:v', str(filter_file)]
    command += ['-fps_mode', 'passthrough', '-f', 'framemd5', '-']
    return [line.rsplit(',', 1)[-1].strip() for line in run(command).splitlines()
            if line and not line.startswith('#')]


def frame_times(path):
    data = json.loads(run(['ffprobe', '-threads', '0', '-v', 'error', '-select_streams', 'v:0',
                          '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time',
                          '-of', 'json', str(path)]))
    return [float(frame['best_effort_timestamp_time']) for frame in data['frames']]


def annotation_hashes(folder):
    files = sorted(folder.rglob('*'))
    if folder.is_symlink() or any(p.is_symlink() for p in files):
        raise ValueError('Annotation symlinks are not supported')
    return {str(p.relative_to(folder)): digest(p) for p in files if p.is_file()}


def source_signature(video, folder, thickness):
    """Content hashes detect edits even if timestamps or file sizes are unchanged."""
    assets = annotation_hashes(folder)
    return {
        'video_sha256': digest(video),
        'sidecar_sha256': digest(video.with_suffix('.json')),
        'annotation_assets_sha256': assets,
        'annotation_sha256': {name: value for name, value in assets.items() if '/' not in name and name.endswith('.json')},
        'box_color': COLOR,
        'box_px': thickness,
        'writer_sha256': digest(Path(__file__)),
        'source_video': str(video.resolve()),
        'source_annotations': str(folder.resolve()),
    }


def output_is_current(destination, signature):
    try:
        paths = output_paths(destination, Path(signature['source_annotations']).name)
        saved = read_json(paths['provenance'])
        previous = dict(saved.get('source_signature', {}))
        if previous.get('writer_sha256') in COMPATIBLE_FLAT_WRITERS:
            previous['writer_sha256'] = signature['writer_sha256']
        if previous != signature:
            return False
        if digest(destination) != saved.get('output_sha256'):
            return False
        return (digest(paths['sidecar']) == signature['sidecar_sha256'] and
                annotation_hashes(paths['annotations']) == signature['annotation_assets_sha256'])
    except (OSError, ValueError, AttributeError, TypeError):
        return False


@contextmanager
def publication_lock(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name('.' + destination.name + '.lock')
    try:
        lock.mkdir()
    except FileExistsError:
        raise ValueError(f'Clip is locked by another or interrupted writer: {lock}') from None
    try:
        yield
    finally:
        lock.rmdir()


def validate_destinations(destination, clip_name):
    if destination.parent.is_symlink():
        raise ValueError('Clip folder must not be a symlink')
    gt = destination.parent / 'GT'
    if gt.is_symlink() or (gt.exists() and not gt.is_dir()):
        raise ValueError('GT must be a directory, not a symlink or file')
    root = destination.parent.resolve()
    paths = output_paths(destination, clip_name)
    for key, path in paths.items():
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe output path: {path}')
        if path.exists() and (path.is_dir() != (key == 'annotations')):
            raise ValueError(f'Unexpected output type: {path}')
    return paths


def publish(staging, destination, clip_name):
    """Publish verified files with rollback; provenance is the final commit marker."""
    paths = validate_destinations(destination, clip_name)
    staged = {'video': 'video.mp4', 'sidecar': 'sidecar.json',
              'annotations': 'annotations', 'provenance': 'provenance.json'}
    backup = destination.parent / ('.backup-' + destination.stem + '-' + uuid.uuid4().hex)
    backup.mkdir()
    (destination.parent / 'GT').mkdir(exist_ok=True)
    moved, installed = [], []
    try:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.rename(backup / key)
                moved.append(key)
            (staging / staged[key]).rename(path)
            installed.append(key)
    except BaseException:
        for key in reversed(installed):
            paths[key].rename(staging / staged[key])
        for key in reversed(moved):
            (backup / key).rename(paths[key])
        backup.rmdir()
        raise
    shutil.rmtree(backup)


def migrate_flat_output(destination, clip_name):
    """Move a verified flat output without decoding, re-encoding, or replacing GT."""
    old = flat_output_paths(destination, clip_name)
    new = validate_destinations(destination, clip_name)
    if destination.exists() or not old['video'].exists():
        return False
    dataset = destination.parent.parent.resolve()
    for key, path in old.items():
        if path.is_symlink() or not path.resolve().is_relative_to(dataset):
            raise ValueError(f'Unsafe legacy output path: {path}')
        if not (path.is_dir() if key == 'annotations' else path.is_file()):
            raise ValueError(f'Incomplete legacy output: {path}')
    if any(path.exists() for path in new.values()):
        raise ValueError('Partial clip folder exists; cannot merge legacy outputs automatically')
    with publication_lock(old['video']):
        saved = read_json(old['provenance'])
        signature = saved.get('source_signature', {})
        if saved.get('clip') != clip_name:
            raise ValueError('Legacy output annotation identity differs')
        if (digest(old['video']) != saved.get('output_sha256') or
                digest(old['sidecar']) != signature.get('sidecar_sha256') or
                annotation_hashes(old['annotations']) != signature.get('annotation_assets_sha256')):
            raise ValueError('Legacy output differs from saved hashes; keeping it in place')
        legacy_gt = dataset / 'cot' / destination.stem
        gt = destination.parent / 'GT'
        if legacy_gt.exists() or legacy_gt.is_symlink():
            if (legacy_gt.is_symlink() or not legacy_gt.is_dir() or
                    not legacy_gt.resolve().is_relative_to(dataset) or gt.exists()):
                raise ValueError('Cannot safely move existing CoT results into GT')
            old['gt'], new['gt'] = legacy_gt, gt
        moved = []
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            for key in old:
                old[key].rename(new[key])
                moved.append(key)
            gt.mkdir(exist_ok=True)
        except BaseException:
            for key in reversed(moved):
                new[key].rename(old[key])
            raise
    return True


def render(video, destination, folder, stream, filters, provenance, signature, before_publish=None):
    if stream['pix_fmt'] not in ('yuv420p', 'yuv422p', 'yuv444p', 'gbrp'):
        raise ValueError(f"Lossless H.264 output is not configured for {stream['pix_fmt']}")
    encoder = 'libx264rgb' if stream['pix_fmt'] == 'gbrp' else 'libx264'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.writing-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'clip'
        staging.mkdir()
        filter_file = Path(temporary) / 'overlay.txt'
        filter_file.write_text(filters)
        output = staging / 'video.mp4'
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(video),
                   '-map', '0:v:0', '-map', '0:a?', '-map_metadata', '0',
                   '-filter_script:v', str(filter_file), '-fps_mode', 'passthrough',
                   '-c:v', encoder, '-qp', '0', '-preset', 'fast', '-bf', '0',
                   '-enc_time_base', '1:1000000', '-video_track_timescale', '1000000',
                   '-movflags', '+faststart',
                   '-pix_fmt', stream['pix_fmt'], '-c:a', 'copy']
        for source_key, flag in [('color_range', '-color_range'), ('color_space', '-colorspace'),
                                 ('color_transfer', '-color_trc'), ('color_primaries', '-color_primaries')]:
            if stream.get(source_key) not in (None, 'unknown', 'unspecified'):
                command += [flag, str(stream[source_key])]
        run(command + [str(output)])
        encoded = probe(output)
        for key in ('width', 'height', 'frames', 'pix_fmt'):
            if stream[key] != encoded[key]:
                raise ValueError(f'Output validation failed: {key} changed')
        if frame_hashes(video, filter_file) != frame_hashes(output):
            raise ValueError('Lossless verification failed: decoded frame hashes differ')
        before, after = frame_times(video), frame_times(output)
        if len(before) != len(after) or any(abs(a - b) > 0.000001 for a, b in zip(before, after)):
            raise ValueError('Output frame timestamps changed by more than 1 microsecond')
        annotations = staging / 'annotations'
        shutil.copytree(folder, annotations)
        shutil.copy2(video.with_suffix('.json'), staging / 'sidecar.json')
        copied_hashes = {p.name: digest(p) for p in sorted(annotations.glob('*.json'))}
        if source_signature(video, folder, provenance['box_px']) != signature:
            raise ValueError('Source changed during rendering; keeping previous output. Run again.')
        if (annotation_hashes(annotations) != signature['annotation_assets_sha256'] or
                digest(staging / 'sidecar.json') != signature['sidecar_sha256']):
            raise ValueError('Copied annotations differ from validated source; keeping previous output')
        provenance.update(source_signature=signature, source_sha256=signature['video_sha256'], output_sha256=digest(output),
                          annotation_sha256=copied_hashes, sidecar_sha256=signature['sidecar_sha256'], codec='h264', container='mp4',
                          encoder=encoder, qp=0, pixel_format=encoded['pix_fmt'],
                          frame_timestamp_tolerance_s=0.000001,
                          verification='decoded frame hashes match uncompressed filtered source')
        provenance['inspection_snapshot'] = snapshot(video, folder, output, annotations, staging / 'sidecar.json')
        (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        if before_publish:
            before_publish()
        publish(staging, destination, folder.name)


def print_run_summary(report, manifest):
    statuses = [('written', '✅ Created'), ('rewritten', '✅ Rewritten'),
                ('unchanged', '⏭️ Unchanged'), ('rejected', '🚫 Rejected'),
                ('ineligible', '⏭️ Ineligible'), ('error', '❌ Failed'),
                ('not_selected_limit', '⏭️ Outside --limit'), ('cancelled', '⏹️ Cancelled')]
    print('\nVideo preprocessing summary')
    for variant in dict.fromkeys(record['variant'] for record in report['clips']):
        counts = Counter(record['status'] for record in report['clips'] if record['variant'] == variant)
        print(f"\n{variant.capitalize()} — processed: {counts['written'] + counts['rewritten']}")
        for status, label in statuses:
            print(f'  {label}: {counts[status]}')
    counts = report['counts']
    print(f"\n✅ Total processed: {counts.get('written', 0) + counts.get('rewritten', 0)} "
          '(created + rewritten)')
    reasons = Counter(record.get('reason', 'Reason unavailable') for record in report['clips']
                      if record['status'] == 'ineligible')
    if reasons:
        print('\nIneligible reasons:')
        for reason, count in sorted(reasons.items()):
            print(f'  {count}: {reason}')
    failures = [record for record in report['clips'] if record['status'] == 'error']
    if failures:
        print('\nFailed clips:')
        for record in failures:
            print(f"  ❌ {record['variant']}/{record['clip']}: {record['reason']}")
    print(f'\nDetailed JSON report: {manifest}')


class NoLongerCompleted(ValueError):
    pass


def process_clip(record, folder, source, output_root, output_name, protected_roots, args, entry=None):
    """Validate and publish one clip; shared report updates stay on the main thread."""
    record = dict(record)
    def check_completion():
        if entry is not None and not args.completion_manifest.is_current(entry, record['variant']):
            raise NoLongerCompleted('Clip removed or changed in the current completion manifest')
    try:
        check_completion()
        meta = read_json(folder / 'meta.json')
        if entry is not None and Path(meta['video_path']).name != entry['video'].name:
            raise ValueError('Export video_path differs from current completion manifest; source export must be refreshed')
        video = source / Path(meta['video_path']).name
        if video.is_symlink():
            raise ValueError('Source video must not be a symlink')
        destination = output_root / output_name / video.stem / video.name
        if any(destination.resolve().is_relative_to(p) for p in protected_roots):
            raise ValueError('Destination resolves inside source tree')
        validate_destinations(destination, folder.name)
        record.update(modality=meta.get('modality'), source_video=str(video), destination=str(destination), source_annotations=str(folder),
                      annotation_status=meta.get('annotation_status'), box_color=COLOR, box_px=args.box_px)
        if not args.write:
            state, _ = output_status(video, folder, destination, digest(Path(__file__)), args.box_px)
            if state == 'current_by_metadata':
                check_completion()
                record.update(status='unchanged', verification_mode='file_metadata')
                return record
        signature = source_signature(video, folder, args.box_px)
        if destination.exists() and not args.write and output_is_current(destination, signature):
            check_completion()
            record['status'] = 'unchanged'
            return record
        video, stream, filters, gaps, drawn = inspect_clip(folder, source, args.box_px, args.selection, current_completion=entry is not None)
        if source_signature(video, folder, args.box_px) != signature:
            raise ValueError('Source changed during validation; run again')
        destination = output_root / output_name / video.stem / video.name
        if any(destination.resolve().is_relative_to(p) for p in protected_roots):
            raise ValueError('Destination resolves inside source tree')
        record.update(status='ready', source_video=str(video), destination=str(destination),
                      annotation_status=meta.get('annotation_status'), frames=stream['frames'],
                      overlay_frames=drawn, omitted_boxes=gaps,
                      box_color=COLOR, box_px=args.box_px, source_annotations=str(folder))
        validate_destinations(destination, folder.name)
        check_completion()
        migrate_flat_output(destination, folder.name)
        exists = destination.exists()
        if exists and not args.write and output_is_current(destination, signature):
            record['status'] = 'unchanged'
            return record
        if exists:
            record.update(status='would_rewrite', reason='Forced regeneration requested' if args.write else 'Source, settings, writer, or output contents changed')
        if not args.inspect:
            with publication_lock(destination):
                # Another writer may have finished between validation and lock acquisition.
                exists = destination.exists()
                if exists and not args.write and output_is_current(destination, signature):
                    record['status'] = 'unchanged'
                    return record
                provenance = dict(record)
                provenance.pop('status')
                provenance.pop('reason', None)
                render(video, destination, folder, stream, filters, provenance, signature, before_publish=check_completion)
                record['status'] = 'rewritten' if exists else 'written'
    except NoLongerCompleted as error:
        record.update(status='ineligible', reason=str(error))
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        record.update(status='error', reason=str(error))
    return record


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(os.environ.get('TRIAGE_DATASET_ROOT', str(DEFAULT_ROOT))))
    parser.add_argument('--completed-manifest', type=Path, help='Current site completion CSV; defaults to source-root/00-data_website/medical/completed_clips.csv')
    parser.add_argument('--output-root', type=Path, help='Parent of 06 and 07; defaults to source-root')
    parser.add_argument('--variant', choices=['both', *VARIANTS], default='both')
    parser.add_argument('--selection', choices=['completed', 'all-non-rejected'], default='completed',
                        help='Select completed annotations (default) or all non-rejected clips with usable inputs')
    parser.add_argument('--box-px', type=int, default=2, help='Clip writer outline thickness (default: 2)')
    parser.add_argument('--workers', type=int, default=1, help='Clips processed concurrently (default: 1)')
    parser.add_argument('--threads-per-worker', type=int, help='FFmpeg decode, encode, and filter thread limits; defaults to 1 with parallel workers')
    parser.add_argument('--limit', type=int, help='Maximum eligible clips to validate per variant')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true', help='Force regeneration of every selected eligible clip, including unchanged outputs')
    mode.add_argument('--inspect', action='store_true', help='Summarize created/changed clips by modality and completion without decoding or hashing videos')
    parser.add_argument('--report', type=Path, help='Also save the JSON report at this explicit path')
    args = parser.parse_args()
    if args.box_px <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('box-px and limit must be positive')
    if args.workers <= 0 or (args.threads_per_worker is not None and args.threads_per_worker <= 0):
        parser.error('workers and threads-per-worker must be positive')
    global MEDIA_THREADS
    MEDIA_THREADS = args.threads_per_worker or (1 if args.workers > 1 else 0)
    source_root = args.source_root.resolve()
    output_root = (args.output_root or source_root).resolve()
    for name, _ in VARIANTS.values():
        protected = source_root / name
        if not (protected / '00-medical_annotations').is_dir():
            parser.error(f'Missing annotation directory: {protected}')
        if output_root == protected or protected in output_root.parents:
            parser.error('Output root must not be inside source folder 04 or 05')
        if args.report:
            report_path = args.report.resolve()
            if report_path == protected or protected in report_path.parents:
                parser.error('Report must not overwrite source annotations or videos')
    protected_roots = [(source_root / names[0]).resolve() for names in VARIANTS.values()]
    for _, output_name in VARIANTS.values():
        target = (output_root / output_name).resolve()
        if any(target == p or p in target.parents for p in protected_roots):
            parser.error('Dataset destination resolves inside source folder 04 or 05')
    if args.report and args.report.exists():
        parser.error('Report already exists; choose a new report path')
    args.completion_manifest = None
    if args.selection == 'completed':
        try:
            args.completion_manifest = CompletionManifest(args.completed_manifest or source_root / DEFAULT_MANIFEST, source_root)
        except (OSError, ValueError) as error:
            parser.error(f'Cannot load current completion manifest: {error}')
        summary = args.completion_manifest.summary()
        print(f"Completion manifest: {summary['kinds'].get('clip', 0)} video clips, "
              f"{summary['kinds'].get('still', 0)} stills excluded; updated {summary['modified_at']}", flush=True)
    rejected = set() if args.completion_manifest else rejected_ids(source_root)
    variants = VARIANTS if args.variant == 'both' else {args.variant: VARIANTS[args.variant]}
    if args.inspect:
        report = inspect_inventory(source_root, output_root, variants, rejected, ACCEPTED,
                                   args.selection, args.box_px, Path(__file__), args.limit, args.completion_manifest)
        print_summary(report)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        return 0
    report = {'schema_version': 1, 'mode': 'inspect' if args.inspect else ('force-write' if args.write else 'update'),
              'source_root': str(source_root), 'output_root': str(output_root), 'selection': args.selection,
              'created_at': datetime.now(timezone.utc).isoformat(), 'clips': []}
    if args.completion_manifest:
        report['completion_manifest'] = args.completion_manifest.summary()
    report['limit'] = args.limit
    report['workers'] = args.workers
    report['threads_per_worker'] = MEDIA_THREADS
    jobs = []
    for variant, (source_name, output_name) in variants.items():
        source = source_root / source_name
        inspected = 0
        entries = args.completion_manifest.entries(variant) if args.completion_manifest else []
        current = {entry['annotations']: entry for entry in entries}
        folders = (sorted(current) if args.completion_manifest else
                   sorted((source / '00-medical_annotations').iterdir()))
        for folder in folders:
            if not args.completion_manifest and not folder.is_dir():
                continue
            entry = current.get(folder)
            record = {'variant': variant, 'clip': folder.name, 'pair_id': pair_id(folder.name)}
            report['clips'].append(record)
            if entry:
                record['completion_level'] = entry['level']
                if args.limit is not None and inspected >= args.limit:
                    record['status'] = 'not_selected_limit'
                    continue
                inspected += 1
                record['status'] = 'queued'
                jobs.append((record, folder, source, output_name, entry))
                continue
            try:
                if pair_id(folder.name) in rejected:
                    record['status'] = 'rejected'
                    continue
                if not (folder / 'meta.json').is_file():
                    record.update(status='error' if entry else 'ineligible', reason='No meta.json for listed completed clip' if entry else 'No meta.json')
                    continue
                meta = read_json(folder / 'meta.json')
                record['modality'] = meta.get('modality')
                if args.selection == 'completed' and not entry and meta.get('annotation_status') not in ACCEPTED:
                    record.update(status='ineligible', reason='Annotation status is not completed_minimal or completed')
                    continue
                if not entry:
                    if args.limit is not None and inspected >= args.limit:
                        record['status'] = 'not_selected_limit'
                        continue
                    inspected += 1
                record['status'] = 'queued'
                jobs.append((record, folder, source, output_name, entry))
            except (OSError, ValueError, KeyError, TypeError) as error:
                record.update(status='error', reason=str(error))
    print(f'Processing {len(jobs)} selected clips with {args.workers} workers; '
          f'FFmpeg threads per worker: {MEDIA_THREADS or "auto"}', flush=True)
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {}
    interrupted = False
    try:
        for record, folder, source, output_name, entry in jobs:
            future = executor.submit(process_clip, record, folder, source, output_root,
                                     output_name, protected_roots, args, entry)
            futures[future] = record
        for done, future in enumerate(as_completed(futures), 1):
            record = futures[future]
            record.update(future.result())
            counts = Counter(c['status'] for c in report['clips'])
            print(f"Finished {done}/{len(jobs)} | created {counts['written']} | "
                  f"updated {counts['rewritten']} | unchanged {counts['unchanged']} | "
                  f"failed {counts['error']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print('Stopping: cancelling queued clips and waiting for active clips to finish safely.', flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=interrupted)
        for future, record in futures.items():
            if future.cancelled():
                record.update(status='cancelled', reason='Run interrupted before processing')
            else:
                record.update(future.result())
        if interrupted:
            for record in report['clips']:
                if record['status'] == 'queued':
                    record.update(status='cancelled', reason='Run interrupted before submission')
        report['interrupted'] = interrupted
    report['counts'] = dict(Counter(c['status'] for c in report['clips']))
    try:
        report['prepared_manifests'] = publish_manifests(report, args.completion_manifest, output_root, variants)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report['manifest_error'] = str(error)
        print(f'Prepared dataset manifest was not fully published: {error}', flush=True)
    encoded = json.dumps(report, indent=2) + '\n'
    if not args.inspect:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = output_root / ('media-writer-run-' + uuid.uuid4().hex + '.json')
        manifest.write_text(encoded)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            stream.write(encoded)
    print_run_summary(report, manifest)
    if args.report:
        print(f'Additional JSON report: {args.report}')
    return 130 if interrupted else (1 if report['counts'].get('error') or report.get('manifest_error') else 0)


if __name__ == '__main__':
    sys.exit(main())
