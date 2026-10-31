#!/usr/bin/env python3
"""Generate native RGB scene annotations; dry run unless --generate is supplied."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from dotenv import load_dotenv
import jsonschema
import yaml

from gemini_media import upload_video

REPO = Path(__file__).resolve().parents[2]
TASK = 'native_general_scene_description'
DATASET = '06-native_media_dataset'


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def result_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode so concurrent processes cannot acquire different locks.
    with path.with_suffix('.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another generator is processing this clip') from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_response(response, schema):
    jsonschema.validate(response, schema)
    for key, value in response.items():
        if isinstance(value, str) and not value.strip():
            raise ValueError(f'Generated {key} is empty')


def prepare(clip, prompt_path, settings):
    prompt = yaml.safe_load(prompt_path.read_text())
    annotations = clip / 'all_annotations'
    meta = read_json(annotations / 'meta.json')
    camera = read_json(annotations / 'camera_motion.json')
    counting = read_json(annotations / 'counting.json')
    if (annotations / 'rejected.json').exists():
        raise ValueError('Clip is rejected')
    if meta.get('annotation_status') not in ('completed', 'completed_minimal'):
        raise ValueError('Clip annotations are not complete')
    if meta.get('modality') != 'RGB':
        raise ValueError('Only RGB clips are supported')
    transcript = (meta.get('scene_description') or {}).get('text')
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError('Scene-description transcript is missing')
    viewpoint = camera.get('view_type')
    if not isinstance(viewpoint, str) or not viewpoint.strip():
        raise ValueError('Camera viewpoint is missing')
    metadata = {key: meta[key] for key in
                ('modality', 'duration_s', 'time_of_day')}
    for key in ('duration_s', 'fps'):
        if not isinstance(meta[key], (int, float)) or not math.isfinite(meta[key]) or meta[key] <= 0:
            raise ValueError(f'Invalid {key}')
    counts = {key: counting[key] for key in
              ('person_count', 'subject_count', 'subject_type', 'other_people_count', 'robot_count')}
    for key, value in counts.items():
        if key != 'subject_type' and (type(value) is not int or value < 0):
            raise ValueError(f'Invalid {key}')
    counting_input = {
        'scope': 'recorded_clip_totals',
        'scope_description': 'Totals recorded for the clip, not counts at a specific timestamp '
                             'or counts of entities visible simultaneously.',
        'totals': counts,
    }
    values = {'metadata_json': json.dumps(metadata, ensure_ascii=False),
              'camera_viewpoint': viewpoint,
              'counting_annotations_json': json.dumps(counting_input, ensure_ascii=False),
              'transcription': transcript}
    text = prompt['user_prompt'].format_map(values)
    video = clip / (clip.name + '.mp4')
    if Path(meta['video_path']).name != video.name:
        raise ValueError('Annotation video identity differs from clip folder')
    provenance_path = clip / 'provenance.json'
    provenance = read_json(provenance_path)
    if provenance.get('clip') != meta.get('clip') or digest(video.with_suffix('.json')) != provenance.get('sidecar_sha256'):
        raise ValueError('Prepared clip identity or sidecar differs from writer provenance')
    hashes = {p.name: digest(p) for p in sorted(annotations.glob('*.json'))}
    video_hash = digest(video)
    if video_hash != provenance.get('output_sha256') or hashes != provenance.get('annotation_sha256'):
        raise ValueError('Prepared dataset differs from writer provenance; rerun the media writer')
    resolved = dict(settings)
    if resolved['video_fps'] == 'source':
        resolved['video_fps'] = meta['fps']
    inputs = {'video_sha256': video_hash, 'annotation_sha256': hashes,
              'provenance_sha256': digest(provenance_path),
              'prompt_sha256': digest(prompt_path), 'settings': resolved,
              'implementation': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'libraries': {name: version(name) for name in ('litellm', 'google-genai')}}
    return {'signature': fingerprint(inputs), 'inputs': inputs, 'prompt': prompt,
            'text': text, 'settings': resolved, 'video': video}


def is_current(path, prepared):
    try:
        saved = read_json(path)
        if saved['generation']['signature'] != prepared['signature']:
            return False
        validate_response(saved['response'], prepared['prompt']['response_schema'])
        return True
    except (OSError, ValueError, KeyError, TypeError, jsonschema.ValidationError):
        return False


def request_payload(prepared, uri):
    """Build the same LiteLLM request for generation and read-only review."""
    settings = prepared['settings']
    # Keep the task prompt and repeated media before clip-specific annotation text.
    messages = [
        {'role': 'system', 'content': prepared['prompt']['system_prompt']},
        {'role': 'user', 'content': [
            {'type': 'file', 'file': {'file_id': uri, 'format': 'video/mp4',
                                    'detail': settings['media_resolution'],
                                    'video_metadata': {'fps': settings['video_fps']}}},
            {'type': 'text', 'text': prepared['text']},
        ]},
    ]
    return {
        'model': settings['model'], 'messages': messages,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': TASK, 'strict': True, 'schema': prepared['prompt']['response_schema']}},
        'reasoning_effort': settings['reasoning_effort'], 'max_tokens': settings['max_tokens'],
        'timeout': 300, 'num_retries': 2,
    }


def generate(prepared, cache_dir):
    import litellm
    uri = upload_video(prepared['video'], prepared['inputs']['video_sha256'], cache_dir)
    response = litellm.completion(**request_payload(prepared, uri))
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        raise ValueError(f'Generation did not finish: {choice.finish_reason}')
    result = json.loads(choice.message.content)
    validate_response(result, prepared['prompt']['response_schema'])
    return result, response.usage.model_dump(mode='json') if response.usage else {}, response.id


def effective_workers(requested, clip_count):
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)
        cpus = cpu_count() or 1
    return min(requested, max(1, cpus), clip_count)


def process_clip(clip, prompt_path, settings, cache, *, write, inspect):
    inspection = None
    try:
        prepared = prepare(clip, prompt_path, settings)
        output = clip / 'GT' / 'general_scene_description.json'
        if inspect:
            inspection = {'clip': clip.name, 'system_prompt': prepared['prompt']['system_prompt'],
                          'user_prompt': prepared['text'], 'settings': prepared['settings'],
                          'response_schema': prepared['prompt']['response_schema']}
        status = 'current' if is_current(output, prepared) else 'would_generate'
        if write and status != 'current':
            with result_lock(output):
                if is_current(output, prepared):
                    status = 'current'
                else:
                    result, usage, request_id = generate(prepared, cache)
                    if prepare(clip, prompt_path, settings)['signature'] != prepared['signature']:
                        raise ValueError('Inputs changed during generation; existing result was preserved')
                    atomic_json(output, {'response': result, 'generation': {
                        'task': TASK, 'prompt_version': prepared['prompt']['version'],
                        'signature': prepared['signature'], 'inputs': prepared['inputs'],
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'usage': usage, 'request_id': request_id}})
                    status = 'generated'
        return inspection, {'clip': clip.name, 'status': status, 'output': str(output)}
    except Exception as error:
        # API exceptions can contain request bodies or credentials; avoid echoing them.
        message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        return inspection, {'clip': clip.name, 'status': 'error', 'error': message}


def generation_settings(video_fps):
    fps = 'source' if video_fps == 'source' else float(video_fps)
    if fps != 'source' and (not math.isfinite(fps) or fps <= 0):
        raise ValueError('FPS must be positive and finite')
    settings = {'model': os.getenv('GEMINI_MODEL', 'gemini/gemini-4-argon'),
                'video_fps': fps,
                'media_resolution': os.getenv('GEMINI_MEDIA_RESOLUTION', 'high'),
                'reasoning_effort': os.getenv('GEMINI_REASONING_EFFORT', 'high'),
                'max_tokens': int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '8192'))}
    if not settings['model'].startswith('gemini/'):
        raise ValueError('GEMINI_MODEL must use the gemini/ provider route')
    if settings['media_resolution'] not in ('low', 'medium', 'high') or settings['reasoning_effort'] not in ('low', 'medium', 'high') or settings['max_tokens'] <= 0:
        raise ValueError('Invalid media resolution, reasoning effort, or output token limit')
    return settings


def main():
    load_dotenv(REPO / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=(os.getenv('AIRLAB_DATASET_ROOT') or os.getenv('TRIAGE_DATASET_ROOT')))
    parser.add_argument('--clip', help='Prepared clip ID; omit to process all prepared native clips')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent clips (default: 4), capped by available CPUs and selected clips')
    parser.add_argument('--generate', action='store_true', help='Upload media, call Gemini, and save results')
    parser.add_argument('--inspect', action='store_true', help='Print rendered prompt and response schema')
    parser.add_argument('--video-fps', default=os.getenv('GEMINI_VIDEO_FPS', 'source'),
                        help='Requested sampling FPS, or source (default); not a guarantee of visual fidelity')
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error('--workers must be positive')
    if args.dataset_root is None:
        parser.error('Set TRIAGE_DATASET_ROOT or provide --dataset-root')
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    if args.clip and Path(args.clip).name != args.clip:
        parser.error('--clip must be a clip ID, not a path')
    try:
        settings = generation_settings(args.video_fps)
    except ValueError as error:
        parser.error(str(error))
    if args.generate and not os.getenv('GEMINI_API_KEY'):
        parser.error('GEMINI_API_KEY is required for --generate')
    dataset = args.dataset_root.expanduser().resolve() / DATASET
    clips = dataset
    if not clips.is_dir():
        parser.error(f'Prepared dataset is missing: {clips}. Run the media writer first.')
    selected = sorted(p for p in clips.iterdir() if p.is_dir() and not p.name.startswith('.')
                      and (p / 'all_annotations').is_dir())
    if args.clip:
        selected = [p for p in selected if p.name == args.clip]
    if not args.clip:
        candidates = []
        for clip in selected:
            try:
                modality = read_json(clip / 'all_annotations' / 'meta.json').get('modality')
            except (OSError, ValueError, AttributeError):
                modality = None  # Report malformed inputs through the per-clip error path.
            if modality in (None, 'RGB'):
                candidates.append(clip)
        selected = candidates
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        parser.error('No prepared RGB clips found')
    prompt_path = Path(__file__).parent / 'yaml' / 'RGB' / f'{TASK}.yaml'
    cache = Path(os.getenv('COT_CACHE_DIR', str(Path.home() / '.cache' / 'triage-cot'))).expanduser()
    workers = effective_workers(args.workers, len(selected))
    print(f'Selected clips: {len(selected)} | Workers: {workers}', file=sys.stderr, flush=True)
    failures = 0
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cot')
    try:
        futures = [executor.submit(process_clip, clip, prompt_path, settings, cache,
                                   write=args.generate, inspect=args.inspect)
                   for clip in selected]
        for future in as_completed(futures):
            inspection, report = future.result()
            if inspection is not None:
                print(json.dumps(inspection, ensure_ascii=False), flush=True)
            failed = report['status'] == 'error'
            failures += int(failed)
            print(json.dumps(report), file=sys.stderr if failed else sys.stdout, flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
