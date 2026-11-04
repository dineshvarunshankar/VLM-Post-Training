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
from itertools import zip_longest
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
import time

from dotenv import load_dotenv
import jsonschema

from transport_media import PROFILE as TRANSPORT_PROFILE, transport_cache
from gemini_media import upload_video
from evidence_media import evidence_images, frame_reference, video_duration
from medical import (CATALOG, TEMPLATES, categories, medical_inputs, validate_medical,
                     scene_medical_context, reference_transcript)
from task_identity import VERSION as TASK_IDENTITY_VERSION, task_input_hash, same_task_inputs, legacy_current_inputs
from quality import review_response
from request_runtime import (REQUESTS, InvalidResponse, GatewayUnavailable, gateway_health,
                             last_metrics, reset_metrics, safe_error_details)
from source_cache import (cached_files, caching_sources, clip_sources, digest, file_bytes,
                          read_json as source_json, read_yaml as source_yaml,
                          verify_sources, watch_sources)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocess.prepared_manifest import DATASETS, PreparedManifest

REPO = Path(__file__).resolve().parents[2]
TASK = 'native_general_scene_description'
DATASET = '06-native_media_dataset'
TASKS = {f'{variant}_{task}': variant for variant in ('stabilized', 'native')
         for task in ('general_scene_description', 'medical_spatiotemporal_vqa')}


def task_batches(tasks):
    """Share native workers while preserving the selected variant order."""
    batches = []
    for task in tasks:
        if batches and TASKS[task] == 'native' and TASKS[batches[-1][0]] == 'native':
            batches[-1].append(task)
        else:
            batches.append([task])
    return batches


def read_json(path):
    return json.loads(path.read_text())


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
    verify_sources()
    prompt = source_yaml(prompt_path)
    task = prompt['name']
    variant = TASKS[task]
    manifest_paths = [clip.parent.parent / folder / 'dataset_manifest.json' for folder in DATASETS.values()]
    def load_manifest():
        if caching_sources():
            for path in manifest_paths:
                file_bytes(path)
        return PreparedManifest(clip.parent.parent)
    manifest = cached_files(manifest_paths, 'manifest', load_manifest)
    entry = next((e for e in manifest.entries(variant) if e['clip_id'] == clip.name), None)
    if entry is None or clip.parent.name != DATASETS[variant] or clip.is_symlink():
        raise ValueError('Clip is not in the prepared manifest for this task')
    medical = 'medical_spatiotemporal' in task
    annotations = clip / 'all_annotations'
    watch_sources([annotations, annotations / 'rejected.json'])
    meta = source_json(annotations / 'meta.json')
    camera = source_json(annotations / 'camera_motion.json')
    counting = source_json(annotations / 'counting.json') if variant == 'native' else None
    # Eligibility comes from the published manifest. A retained rejected.json
    # can predate a later completion; it remains part of the provenance checks.
    if meta.get('annotation_status') not in ('completed', 'completed_minimal'):
        raise ValueError('Clip annotations are not complete')
    if meta.get('modality') != 'RGB':
        raise ValueError('Only RGB clips are supported')
    transcript = reference_transcript((meta.get('scene_description') or {}).get('text'))
    viewpoint = camera.get('view_type')
    if not isinstance(viewpoint, str) or not viewpoint.strip():
        raise ValueError('Camera viewpoint is missing')
    video = clip / (clip.name + '.mp4')
    duration_missing = meta.get('duration_s') is None
    if duration_missing:
        meta['duration_s'] = video_duration(video)
    for key in ('duration_s', 'fps'):
        if not isinstance(meta[key], (int, float)) or not math.isfinite(meta[key]) or meta[key] <= 0:
            raise ValueError(f'Invalid {key}')
    metadata = {'modality': meta['modality'], 'duration_s': meta['duration_s'],
                'time_of_day': meta.get('time_of_day')}
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
    if Path(meta['video_path']).name != video.name:
        raise ValueError('Annotation video identity differs from clip folder')
    provenance_path = clip / 'provenance.json'
    provenance = source_json(provenance_path)
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
        resolved['video_fps'] = min(meta['fps'], 24.0)
    inputs = {'video_sha256': video_hash, 'annotation_sha256': hashes,
              'provenance_sha256': digest(provenance_path),
              'prompt_sha256': digest(prompt_path), 'settings': resolved,
              'implementation': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
              'libraries': {name: version(name) for name in ('litellm', 'google-genai')}}
    if resolved.get('api_base') and variant == 'native':
        inputs['video_transport'] = dict(TRANSPORT_PROFILE)
    inputs['catalog_sha256'] = digest(CATALOG)
    if duration_missing:
        inputs['resolved_video_duration_s'] = meta['duration_s']
        inputs['duration_source'] = 'prepared_video'
    inputs['assembled_input_sha256'] = fingerprint(values)
    if medical:
        inputs['assessment_prompts_sha256'] = digest(TEMPLATES)
        inputs['category'] = category
    inputs['task_identity_version'] = TASK_IDENTITY_VERSION
    inputs['task_input_sha256'] = task_input_hash(prompt, values, prompt['response_schema'], extra)
    inputs['video_fps_policy'] = settings['video_fps']
    if caching_sources():
        verify_sources()
    else:
        manifest.verify_unchanged()
    return {**extra, 'task': task, 'category': category, 'signature': fingerprint(inputs), 'inputs': inputs, 'prompt': prompt,
            'text': text, 'settings': resolved, 'video': video}


def compatible_inputs(previous, current):
    """Accept the audited preflight/retry-only upgrade, not arbitrary code changes.

    Preserve accepted original-video outputs across the approved CRF 18 transport migration.
    Source, prompt, annotation, and validation changes still invalidate outputs.
    All other implementation files and every input/config hash must still match.
    """
    legacy = legacy_current_inputs(current)
    if legacy is not None and 'task_identity_version' not in previous:
        return previous == legacy or compatible_inputs(previous, legacy)
    prior_schedulers = [{'generate.py': '36b21ca4b989bb8e222c50821f5eed54a69716e37d8eece66891997507acbfbf', 'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25'}, {
        'generate.py': '4fa8af4913131a8c6836d5e30942ff8f7ee9f7694d5c70c887b6aa1d462f5bbd',
        'request_runtime.py': '9fe12217d41ca0690610a1d29435d6e7dd414d17b58e3eb9a043a2db92d69383',
    }, {
        'generate.py': 'ab8117d3dfe41b6b47d3ab7b6645186452ea3be6d7a1e09d816bdfeca3f7a7a3',
        'request_runtime.py': 'e451a35848c4b2fcb0c7766ec0731bfc5ebf14aed1799e88ad100cfb1f6a7d3b',
    }, {
        'generate.py': 'ff05023aea6d274980de8020d725bcf130b0a816752308860d67a0ea6980bca6',
        'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25',
    }, {
        'generate.py': '39dbd2665ec0debe95574071e401b7a6d85f3924c9313f5b75d5aa2a59e10cec',
        'request_runtime.py': '7205b8b4a202a2f3d035bcc589d404ce95cb3cfc9781ecbedc195cad16ce5b25',
    }]
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    old_impl, new_impl = previous.get('implementation', {}), current.get('implementation', {})
    # These audited routing/eligibility fixes leave accepted request content and
    # validation unchanged. Keep every other implementation and input check.
    if old_impl.get('generate.py') in {
        '95bdbfb6970b8f2e21ec7b2208a97b4f206d52bf41f1af945e459c975dcf6b62',
        '70b23a2a44a0ed89e52bd79033fa2ce43f8f7cc18aa8ff1e0f50df9802b0e5fd',
    }:
        return ({k: v for k, v in old_impl.items() if k != 'generate.py'}
                == {k: v for k, v in new_impl.items() if k != 'generate.py'}
                and {k: v for k, v in previous.items() if k != 'implementation'}
                == {k: v for k, v in current.items() if k != 'implementation'})
    if not any(all(old_impl.get(name) == digest for name, digest in profile.items())
               for profile in prior_schedulers):
        return False
    excluded = {'generate.py', 'request_runtime.py', 'preflight.py', 'transport_media.py'}
    return (old_impl.keys() <= new_impl.keys()
            and {k: v for k, v in old_impl.items() if k not in excluded}
            == {k: v for k, v in new_impl.items() if k not in excluded}
            and {k: v for k, v in previous.items() if k != 'implementation'}
            == {k: v for k, v in current.items() if k not in ('implementation', 'video_transport')})


def is_current(path, prepared):
    try:
        saved = read_json(path)
        if saved['generation']['signature'] != prepared['signature']:
            previous = saved['generation'].get('inputs')
            if (not previous or saved['generation']['signature'] != fingerprint(previous)
                    or not (same_task_inputs(previous, prepared['inputs'])
                            or compatible_inputs(previous, prepared['inputs']))):
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
        'timeout': 300, 'num_retries': 0,
    }


def generate(prepared, cache_dir):
    import litellm
    reset_metrics()
    credentials = {}
    if prepared['settings'].get('api_base'):
        def inline_video():
            if TASKS[prepared['task']] == 'stabilized':
                data = file_bytes(prepared['video'])
                video_hash = hashlib.sha256(data).hexdigest()
                if video_hash != prepared['inputs']['video_sha256']:
                    raise ValueError('Video changed before request')
                metadata = {'source_sha256': video_hash, 'sha256': video_hash,
                            'source_bytes': len(data), 'bytes': len(data),
                            'profile': {'mode': 'original_lossless'}}
                return 'data:video/mp4;base64,' + base64.b64encode(data).decode(), metadata
            asset = transport_cache(cache_dir).get(prepared['video'], prepared['inputs']['video_sha256'])
            data = asset.path.read_bytes()
            if hashlib.sha256(data).hexdigest() != asset.metadata['sha256']:
                raise ValueError('Transport copy changed before request')
            return ('data:video/mp4;base64,' + base64.b64encode(data).decode(), asset.metadata)
        uri, transport = cached_files([prepared['video']],
                                      ('transport_video', prepared['inputs']['video_sha256']), inline_video)
        prepared['transport'] = transport
        credentials['api_key'] = os.environ['GEMINI_API_KEY']
    else:
        uri = upload_video(prepared['video'], prepared['inputs']['video_sha256'], cache_dir)
    images = evidence_images(prepared)
    def call(**kwargs):
        verify_sources()
        return litellm.completion(**kwargs, **credentials)
    result = None
    def validate(response):
        nonlocal result
        try:
            choice = response.choices[0]
            if choice.finish_reason != 'stop':
                raise InvalidResponse(f'Generation did not finish: {choice.finish_reason}')
            result = json.loads(choice.message.content)
            validate_response(result, prepared['prompt']['response_schema'])
            if 'expected_answer' in prepared:
                validate_medical(result, prepared)
                result['answer'] = dict(prepared['expected_answer'])
        except jsonschema.ValidationError:
            raise InvalidResponse('Response does not match the required JSON schema') from None
        except (ValueError, TypeError, KeyError, IndexError) as error:
            raise InvalidResponse(str(error)[:500]) from error
    response = REQUESTS.complete(call, request_payload(prepared, uri, images), validate=validate)
    return result, response.usage.model_dump(mode='json') if response.usage else {}, response.id


def effective_workers(requested, clip_count):
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)
        cpus = cpu_count() or 1
    return min(requested, max(1, cpus), clip_count)


def process_clip(clip, prompt_path, settings, cache, *, write, inspect, category=None, expected_signature=None):
    reset_metrics()
    inspection = None
    review = None
    try:
        if write:
            REQUESTS.raise_if_unavailable()
        prepared = prepare(clip, prompt_path, settings, category)
        if expected_signature is not None and prepared['signature'] != expected_signature:
            raise ValueError('Inputs changed after preflight; rerun preflight before generation')
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
                    verify_sources()
                    review = review_response(result)
                    atomic_json(output, {'response': result, 'review': review, 'generation': {
                        'task': prepared['task'], 'category': category, 'prompt_version': prepared['prompt']['version'],
                        'signature': prepared['signature'], 'inputs': prepared['inputs'],
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'usage': usage, 'request_id': request_id, 'request_metrics': last_metrics(),
                        **({'transport': prepared['transport']} if 'transport' in prepared else {})}})
                    status = 'generated'
        if status == 'current':
            review = review_response(read_json(output)['response'])
        return inspection, {'clip': clip.name, 'category': category, 'status': status, 'output': str(output),
                            'review': review, 'context_exclusions': prepared.get('context_exclusions', {}),
                            **({'request_metrics': last_metrics()} if last_metrics() else {})}
    except GatewayUnavailable:
        raise
    except Exception as error:
        # API exceptions can contain request bodies or credentials; avoid echoing them.
        message = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        return inspection, {'clip': clip.name, 'category': category, 'status': 'error', 'error': message,
                            **({'error_details': safe_error_details(error)} if getattr(error, 'status_code', None) else {}),
                            **({'request_metrics': last_metrics()} if last_metrics() else {})}


def generation_settings(video_fps):
    fps = 'source' if video_fps == 'source' else float(video_fps)
    if fps != 'source' and (not math.isfinite(fps) or fps <= 0):
        raise ValueError('FPS must be positive and finite')
    if fps != 'source' and fps > 24:
        raise ValueError('Gemini sampling FPS must not exceed 24; use source for automatic capping')
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
                                help='Task names in execution order, or all; adjacent native tasks share workers')
    parser.add_argument('--category', help='Medical category key; omit to process all valid categories')
    parser.add_argument('--clip', help='Prepared clip ID; omit to use the prepared manifest')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4,
                        help='Concurrent clips (default: 4), capped by available CPUs and selected clips')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--generate', action='store_true', help='Upload media, call Gemini, and save results')
    mode.add_argument('--inspect', action='store_true', help='Print rendered prompt and response schema')
    mode.add_argument('--preflight', action='store_true', help='Check all selected inputs and save a report; no API calls')
    parser.add_argument('--skip-invalid', action='store_true',
                        help='Generate valid tasks/categories while reporting blocked inputs')
    parser.add_argument('--report', type=Path, help='Preflight report path (default: .local-reports/cot/)')
    parser.add_argument('--video-fps', default=os.getenv('GEMINI_VIDEO_FPS', 'source'),
                        help='Sampling FPS up to 24, or source capped at 24 (default)')
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
    if args.generate:
        health = gateway_health(settings.get('api_base'), settings['model'], os.getenv('GEMINI_API_KEY'))
        if not health['healthy']:
            print(f'CoT deferred: {health["reason"]}. No generation requests made.', file=sys.stderr, flush=True)
            return 75
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
    plan = None
    failures = 0
    if args.generate or args.preflight:
        from preflight import build_plan
        plan = build_plan(tasks, selected_by_variant, args, settings, engine=sys.modules[__name__])
        manifest.verify_unchanged()
        plan['prepared_manifests'] = manifest.summary()
        report = args.report or (REPO / '.local-reports/cot' /
                                ('preflight-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'))
        atomic_json(report, plan)
        print(f'Preflight report: {report}', file=sys.stderr, flush=True)
        failures = int(bool(plan['errors']))
        if args.preflight:
            return failures
        if failures and not args.skip_invalid:
            print('Preflight blocked generation; no API calls made. Correct the reported inputs, '
                  'or use --skip-invalid to process valid categories.', file=sys.stderr, flush=True)
            return 1
    if args.generate and settings.get('api_base'):
        cache = transport_cache(Path(os.getenv('COT_CACHE_DIR', str(Path.home() / '.cache' / 'triage-cot'))))
        for variant, clips in selected_by_variant.items():
            for clip in clips:
                pending_tasks = {task for task in tasks if TASKS[task] == variant
                                 and str(clip) in plan['tasks'][task]['jobs']}
                cache.register(clip, pending_tasks)
    try:
        for batch in task_batches(tasks):
            failures |= run_task_group(batch, selected_by_variant[TASKS[batch[0]]], args, settings,
                                       plans=plan['tasks'] if plan is not None else None)
    except GatewayUnavailable as error:
        print(f'CoT stopped: {error}. Saved outputs retained; pending work will retry on a later run.',
              file=sys.stderr, flush=True)
        return 75
    return int(bool(failures))


def run_task(task, selected, args, settings, *, plan=None):
    return run_task_group([task], selected, args, settings,
                          plans={task: plan} if plan is not None else None)


def run_task_group(tasks, selected, args, settings, *, plans=None):
    """Interleave task/clip jobs in one bounded pool; cache categories per clip."""
    started = time.monotonic()
    cache = Path(os.getenv('COT_CACHE_DIR', str(Path.home() / '.cache' / 'triage-cot'))).expanduser()
    selected_count = len(selected)
    queues, counts = [], {}
    for task in tasks:
        plan = plans[task] if plans is not None else None
        pending = [clip for clip in selected if plan is None or str(clip) in plan['jobs']]
        queues.append([(task, clip) for clip in pending])
        counts[task] = {'finished': 0, 'generated': 0, 'current': 0, 'would_generate': 0, 'error': 0}
        print(f'Task: {task} | Selected clips: {selected_count} | Pending clips: {len(pending)}', file=sys.stderr, flush=True)
        if plan is not None:
            counts[task]['current'] = counts[task]['finished'] = plan['counts']['current']
            print(f'Preflight reused: {counts[task]["current"]} | Blocked inputs: {plan["counts"]["blocked"]}',
                  file=sys.stderr, flush=True)
    jobs = [job for row in zip_longest(*queues) for job in row if job is not None]
    workers = effective_workers(args.workers, len(jobs))
    print(f'Shared workers: {workers} | Tasks: {", ".join(tasks)}', file=sys.stderr, flush=True)
    failures = 0
    if not jobs:
        print(f'Tasks finished: {", ".join(tasks)} | No pending valid results', file=sys.stderr, flush=True)
        return 0
    report_lock = Lock()
    def process_tasks(task, clip):
        variant = TASKS[task]
        prompt_path = Path(__file__).parent / 'yaml' / 'RGB' / f'{task}.yaml'
        plan = plans[task] if plans is not None else None
        if plan is not None:
            for job in plan['jobs'][str(clip)]:
                yield process_clip(clip, prompt_path, settings, cache, write=args.generate,
                                   inspect=args.inspect, category=job['category'],
                                   expected_signature=job['signature'])
            return
        names = [None]
        if 'medical_spatiotemporal' in task:
            try:
                names, excluded = categories(clip, variant)
                if args.category:
                    names, excluded = [args.category], {}
                for name, error in excluded.items():
                    yield None, {'clip': clip.name, 'category': name, 'status': 'error', 'error': error}
            except (OSError, ValueError, KeyError) as error:
                yield None, {'clip': clip.name, 'status': 'error', 'error': str(error)}
                return
        for category in names:
            yield process_clip(clip, prompt_path, settings, cache, write=args.generate,
                               inspect=args.inspect, category=category)
    def process_and_report(task, clip):
        nonlocal failures
        success = True
        completed = False
        try:
            with clip_sources():
                for inspection, report in process_tasks(task, clip):
                    with report_lock:
                        report = {'task': task, **report}
                        if inspection is not None:
                            print(json.dumps(inspection, ensure_ascii=False), flush=True)
                        failed = report['status'] == 'error'
                        success = success and not failed
                        failures += int(failed)
                        count = counts[task]
                        count['finished'] += 1
                        count[report['status']] += 1
                        print(json.dumps(report), file=sys.stderr if failed else sys.stdout, flush=True)
                        print(f"Results {task}: {count['finished']} | generated {count['generated']} | "
                              f"unchanged {count['current']} | pending {count['would_generate']} | "
                              f"failed {count['error']}", file=sys.stderr, flush=True)
            completed = True
        finally:
            if args.generate and settings.get('api_base'):
                transport_cache(cache).finish(clip, task, success and completed)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cot')
    try:
        futures = [executor.submit(process_and_report, task, clip) for task, clip in jobs]
        for future in as_completed(futures):
            future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    elapsed = time.monotonic() - started
    for task, count in counts.items():
        print(f'Task finished: {task} | elapsed {elapsed:.1f}s | '
              f'generated/min {count["generated"] * 60 / max(elapsed, 0.001):.2f}', file=sys.stderr, flush=True)
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
