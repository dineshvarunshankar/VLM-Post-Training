#!/usr/bin/env python3
"""Generate RGB scene descriptions or grounded medical reasoning; dry run by default."""
import argparse
import base64
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
from evidence_media import evidence_images, frame_reference
from medical import CATALOG, TEMPLATES, categories, medical_inputs, validate_medical, scene_medical_context
from quality import review_response
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocess.prepared_manifest import DATASETS, PreparedManifest

REPO = Path(__file__).resolve().parents[2]
TASK = 'native_general_scene_description'
DATASET = '06-native_media_dataset'
TASKS = {f'{variant}_{task}': variant for variant in DATASETS
         for task in ('general_scene_description', 'medical_spatiotemporal_vqa')}


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


def prepare(clip, prompt_path, settings, category=None):
    prompt = yaml.safe_load(prompt_path.read_text())
    task = prompt['name']
    variant = TASKS[task]
    manifest = PreparedManifest(clip.parent.parent)
    entry = next((e for e in manifest.entries(variant) if e['clip_id'] == clip.name), None)
    if entry is None or clip.parent.name != DATASETS[variant] or clip.is_symlink():
        raise ValueError('Clip is not in the prepared manifest for this task')
    medical = 'medical_spatiotemporal' in task
    annotations = clip / 'all_annotations'
    meta = read_json(annotations / 'meta.json')
    camera = read_json(annotations / 'camera_motion.json')
    counting = read_json(annotations / 'counting.json') if variant == 'native' else None
    if (annotations / 'rejected.json').exists():
        raise ValueError('Clip is rejected')
    if meta.get('annotation_status') not in ('completed', 'completed_minimal'):
        raise ValueError('Clip annotations are not complete')
    if meta.get('modality') != 'RGB':
        raise ValueError('Only RGB clips are supported')
    transcript = (meta.get('scene_description') or {}).get('text')
    if not medical and (not isinstance(transcript, str) or not transcript.strip()):
        raise ValueError('Scene-description transcript is missing')
    viewpoint = camera.get('view_type')
    if not isinstance(viewpoint, str) or not viewpoint.strip():
        raise ValueError('Camera viewpoint is missing')
    metadata = {key: meta[key] for key in
                ('modality', 'duration_s', 'time_of_day')}
    for key in ('duration_s', 'fps'):
        if not isinstance(meta[key], (int, float)) or not math.isfinite(meta[key]) or meta[key] <= 0:
            raise ValueError(f'Invalid {key}')
    values = {'metadata_json': json.dumps(metadata, ensure_ascii=False),
              'camera_viewpoint': viewpoint, 'transcription': transcript}
    if counting is not None:
        counts = {key: counting[key] for key in
                  ('person_count', 'subject_count', 'subject_type', 'other_people_count', 'robot_count')}
        for key, value in counts.items():
            if key != 'subject_type' and (type(value) is not int or value < 0):
                raise ValueError(f'Invalid {key}')
        values['counting_annotations_json'] = json.dumps({
            'scope': 'recorded_clip_totals',
            'scope_description': 'Recorded totals for the clip, not counts at a specific timestamp or simultaneous visibility.',
            'totals': counts}, ensure_ascii=False)
    extra = {}
    if medical:
        if not category:
            raise ValueError('Medical tasks require a category')
        additions, schema, extra = medical_inputs(clip, meta, viewpoint, category, variant, prompt['response_schema'])
        values.update(additions)
        prompt['response_schema'] = schema
    else:
        context, excluded = scene_medical_context(clip, meta, variant)
        values['medical_annotations_json'] = json.dumps(context, ensure_ascii=False)
        extra['context_exclusions'] = excluded
    text = prompt['user_prompt'].format_map(values)
    video = clip / (clip.name + '.mp4')
    if Path(meta['video_path']).name != video.name:
        raise ValueError('Annotation video identity differs from clip folder')
    provenance_path = clip / 'provenance.json'
    provenance = read_json(provenance_path)
    if digest(provenance_path) != entry['provenance_sha256'] or meta.get('clip') != entry['clip']:
        raise ValueError('Prepared clip differs from its published manifest')
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
    inputs['catalog_sha256'] = digest(CATALOG)
    inputs['assembled_input_sha256'] = fingerprint(values)
    if medical:
        inputs['assessment_prompts_sha256'] = digest(TEMPLATES)
        inputs['category'] = category
    manifest.verify_unchanged()
    return {**extra, 'task': task, 'category': category, 'signature': fingerprint(inputs), 'inputs': inputs, 'prompt': prompt,
            'text': text, 'settings': resolved, 'video': video}


def is_current(path, prepared):
    try:
        saved = read_json(path)
        if saved['generation']['signature'] != prepared['signature']:
            return False
        validate_response(saved['response'], prepared['prompt']['response_schema'])
        if 'expected_answer' in prepared:
            validate_medical(saved['response'], prepared)
        return True
    except (OSError, ValueError, KeyError, TypeError, jsonschema.ValidationError):
        return False


def request_payload(prepared, uri, image_parts=None):
    """Build the same LiteLLM request for generation and read-only review."""
    settings = prepared['settings']
    # Keep the task prompt and repeated media before clip-specific annotation text.
    messages = [
        {'role': 'system', 'content': prepared['prompt']['system_prompt']},
        {'role': 'user', 'content': [
            {'type': 'file', 'file': {('file_data' if settings.get('api_base') else 'file_id'): uri, 'format': 'video/mp4',
                                    'detail': settings['media_resolution'],
                                    'video_metadata': {'fps': settings['video_fps']}}},
            {'type': 'text', 'text': prepared['text']},
        ]},
    ]
    if image_parts is None:
        image_parts = []
        for frame in prepared.get('evidence_frames', []):
            image_parts.extend([{'type': 'text', 'text': 'Evidence frame: '+json.dumps(frame_reference(frame))},
                                {'type': 'image_url', 'image_url': {'url': "<evidence_image_png>"}}])
    messages[1]['content'].extend(image_parts)
    return {
        'model': ('litellm_proxy/' if settings.get('api_base') else '') + settings['model'],
        **({'api_base': settings['api_base']} if settings.get('api_base') else {}),
        'messages': messages,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': prepared['task'], 'strict': True, 'schema': prepared['prompt']['response_schema']}},
        'reasoning_effort': settings['reasoning_effort'], 'max_tokens': settings['max_tokens'],
        'timeout': 300, 'num_retries': 2,
    }


def generate(prepared, cache_dir):
    import litellm
    credentials = {}
    if prepared['settings'].get('api_base'):
        data = prepared['video'].read_bytes()
        if hashlib.sha256(data).hexdigest() != prepared['inputs']['video_sha256']:
            raise ValueError('Video changed before request')
        uri = 'data:video/mp4;base64,' + base64.b64encode(data).decode()
        credentials['api_key'] = os.environ['GEMINI_API_KEY']
    else:
        uri = upload_video(prepared['video'], prepared['inputs']['video_sha256'], cache_dir)
    response = litellm.completion(**request_payload(prepared, uri, evidence_images(prepared)), **credentials)
    choice = response.choices[0]
    if choice.finish_reason != 'stop':
        raise ValueError(f'Generation did not finish: {choice.finish_reason}')
    result = json.loads(choice.message.content)
    validate_response(result, prepared['prompt']['response_schema'])
    if 'expected_answer' in prepared:
        validate_medical(result, prepared)
        result['answer'] = dict(prepared['expected_answer'])
    return result, response.usage.model_dump(mode='json') if response.usage else {}, response.id


def effective_workers(requested, clip_count):
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)
        cpus = cpu_count() or 1
    return min(requested, max(1, cpus), clip_count)


def process_clip(clip, prompt_path, settings, cache, *, write, inspect, category=None):
    inspection = None
    review = None
    try:
        prepared = prepare(clip, prompt_path, settings, category)
        output = (clip / 'GT' / 'medical_spatiotemporal_vqa' / f'{category}.json' if category
                  else clip / 'GT' / 'general_scene_description.json')
        if output.parent.is_symlink() or output.is_symlink() or not output.resolve().is_relative_to(clip.resolve()):
            raise ValueError('Output path escapes the clip folder')
        if inspect:
            inspection = {'clip': clip.name, 'category': category, 'task': prepared['task'], 'evidence_frames': prepared.get('evidence_frames', []), 'system_prompt': prepared['prompt']['system_prompt'],
                          'user_prompt': prepared['text'], 'settings': prepared['settings'],
                          'response_schema': prepared['prompt']['response_schema'],
                          'context_exclusions': prepared.get('context_exclusions', {})}
        status = 'current' if is_current(output, prepared) else 'would_generate'
        if write and status != 'current':
            with result_lock(output):
                if is_current(output, prepared):
                    status = 'current'
                else:
                    result, usage, request_id = generate(prepared, cache)
                    if prepare(clip, prompt_path, settings, category)['signature'] != prepared['signature']:
                        raise ValueError('Inputs changed during generation; existing result was preserved')
                    review = review_response(result)
                    atomic_json(output, {'response': result, 'review': review, 'generation': {
                        'task': prepared['task'], 'category': category, 'prompt_version': prepared['prompt']['version'],
                        'signature': prepared['signature'], 'inputs': prepared['inputs'],
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'usage': usage, 'request_id': request_id}})
                    status = 'generated'
        if status == 'current':
            review = review_response(read_json(output)['response'])
        return inspection, {'clip': clip.name, 'category': category, 'status': status, 'output': str(output),
                            'review': review, 'context_exclusions': prepared.get('context_exclusions', {})}
    except Exception as error:
        # API exceptions can contain request bodies or credentials; avoid echoing them.
        message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        return inspection, {'clip': clip.name, 'category': category, 'status': 'error', 'error': message}


def generation_settings(video_fps):
    fps = 'source' if video_fps == 'source' else float(video_fps)
    if fps != 'source' and (not math.isfinite(fps) or fps <= 0):
        raise ValueError('FPS must be positive and finite')
    settings = {'model': os.getenv('GEMINI_MODEL', 'gemini/gemini-4-argon'),
                'video_fps': fps,
                'media_resolution': os.getenv('GEMINI_MEDIA_RESOLUTION', 'high'),
                'reasoning_effort': os.getenv('GEMINI_REASONING_EFFORT', 'high'),
                'max_tokens': int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '32768'))}
    api_base = os.getenv('GEMINI_API_BASE', '').strip().rstrip('/')
    if api_base:
        settings['api_base'] = api_base
    if not settings['model'].startswith('gemini/'):
        raise ValueError('GEMINI_MODEL must use the gemini/ provider route')
    if settings['media_resolution'] not in ('low', 'medium', 'high') or settings['reasoning_effort'] not in ('low', 'medium', 'high') or settings['max_tokens'] <= 0:
        raise ValueError('Invalid media resolution, reasoning effort, or output token limit')
    return settings


def main():
    load_dotenv(REPO / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=(os.getenv('AIRLAB_DATASET_ROOT') or os.getenv('TRIAGE_DATASET_ROOT')))
    task_selection = parser.add_mutually_exclusive_group()
    task_selection.add_argument('--task', choices=list(TASKS), help=f'Single task (default: {TASK})')
    task_selection.add_argument('--tasks', nargs='+', choices=['all', *TASKS],
                                help='Task names in execution order, or all; tasks run sequentially')
    parser.add_argument('--category', help='Medical category key; omit to process all valid categories')
    parser.add_argument('--clip', help='Prepared clip ID; omit to use the prepared manifest')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent clips (default: 4), capped by available CPUs and selected clips')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--generate', action='store_true', help='Upload media, call Gemini, and save results')
    mode.add_argument('--inspect', action='store_true', help='Print rendered prompt and response schema')
    parser.add_argument('--video-fps', default=os.getenv('GEMINI_VIDEO_FPS', 'source'),
                        help='Requested sampling FPS, or source (default); not a guarantee of visual fidelity')
    args = parser.parse_args()
    tasks = args.tasks or [args.task or TASK]
    if 'all' in tasks:
        if tasks != ['all']:
            parser.error('Use --tasks all alone, without other task names')
        tasks = list(TASKS)
    tasks = list(dict.fromkeys(tasks))
    if args.category and any('medical_spatiotemporal' not in task for task in tasks):
        parser.error('--category requires selecting only medical tasks')
    if args.clip and len({TASKS[task] for task in tasks}) > 1:
        parser.error('--clip requires tasks from one variant; native and stabilized clip IDs differ')
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
    root = args.dataset_root.expanduser().resolve()
    selected_by_variant = {}
    try:
        manifest = PreparedManifest(root)
        for variant in dict.fromkeys(TASKS[task] for task in tasks):
            selected = []
            for entry in manifest.entries(variant):
                if args.clip and entry['clip_id'] != args.clip:
                    continue
                clip = root / DATASETS[variant] / entry['clip_id']
                if read_json(clip / 'all_annotations/meta.json').get('modality') == 'RGB':
                    selected.append(clip)
            if args.limit:
                selected = selected[:args.limit]
            if not selected:
                raise ValueError(f'No prepared RGB clips found for {variant}')
            selected_by_variant[variant] = selected
        manifest.verify_unchanged()
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for task in tasks:
        if run_task(task, selected_by_variant[TASKS[task]], args, settings):
            return 1
    return 0


def run_task(task, selected, args, settings):
    """Finish one task's clip batch before starting the next task."""
    variant = TASKS[task]
    prompt_path = Path(__file__).parent / 'yaml' / 'RGB' / f'{task}.yaml'
    cache = Path(os.getenv('COT_CACHE_DIR', str(Path.home() / '.cache' / 'triage-cot'))).expanduser()
    workers = effective_workers(args.workers, len(selected))
    print(f'Task: {task} | Selected clips: {len(selected)} | Workers: {workers}', file=sys.stderr, flush=True)
    failures = 0
    def process_tasks(clip):
        names = [None]
        reports = []
        if 'medical_spatiotemporal' in task:
            try:
                names, excluded = categories(clip, variant)
                if args.category:
                    names, excluded = [args.category], {}
                for name, error in excluded.items():
                    reports.append((None, {'clip': clip.name, 'category': name, 'status': 'error', 'error': error}))
            except (OSError, ValueError, KeyError) as error:
                return [(None, {'clip': clip.name, 'status': 'error', 'error': str(error)})]
        for category in names:
            reports.append(process_clip(clip, prompt_path, settings, cache, write=args.generate,
                                        inspect=args.inspect, category=category))
        return reports
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cot')
    try:
        futures = [executor.submit(process_tasks, clip) for clip in selected]
        for future in as_completed(futures):
            for inspection, report in future.result():
                report = {'task': task, **report}
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
