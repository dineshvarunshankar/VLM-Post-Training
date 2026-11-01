#!/usr/bin/env python3
"""Read-only localhost review of CoT prompts and available model inputs."""
import argparse
import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv
from jsonschema.exceptions import ValidationError
import yaml

from generate import DATASET, REPO, TASK, generation_settings, prepare, read_json, request_payload, PreparedManifest, is_current, validate_response
from evidence_media import evidence_images
from medical import categories, validate_medical
from quality import review_response

ASSETS = Path(__file__).parent / 'preview'
PROMPTS = Path(__file__).parent / 'yaml' / 'RGB'
PROMPT = PROMPTS / f'{TASK}.yaml'
TASKS = {
    'native_general_scene_description': 'Native scene description',
    'stabilized_general_scene_description': 'Stabilized scene description',
    'native_medical_spatiotemporal_vqa': 'Native medical spatiotemporal VQA',
    'stabilized_medical_spatiotemporal_vqa': 'Stabilized medical spatiotemporal VQA',
}


def response_checks(response, prepared):
    checks = {'response_schema': 'passed', 'visual_support': 'not_checked',
              'cross_task_consistency': 'not_checked'}
    try:
        validate_response(response, prepared['prompt']['response_schema'])
    except (ValidationError, ValueError, TypeError, KeyError):
        checks['response_schema'] = 'failed'
    if 'expected_answer' in prepared:
        checks['human_answer_and_grounding'] = 'not_checked'
        if checks['response_schema'] == 'passed':
            try:
                validate_medical(response, prepared)
                checks['human_answer_and_grounding'] = 'passed'
            except (ValueError, TypeError, KeyError):
                checks['human_answer_and_grounding'] = 'failed'
    return checks


class Review:
    def __init__(self, root, settings, focus_clips=None, focus_category=None):
        self.root = root.resolve()
        self.settings = settings
        self._index = {}
        self.focus_clips = focus_clips or {}
        self.focus_category = focus_category

    def dataset(self, task):
        if task not in TASKS:
            raise ValueError('Unknown CoT task')
        return self.root / ('07-stabilized_media_dataset' if task.startswith('stabilized') else DATASET)

    def clip(self, name, task=TASK):
        if not name or Path(name).name != name or name.startswith('.'):
            raise ValueError('Invalid clip ID')
        variant = 'stabilized' if task.startswith('stabilized') else 'native'
        if self.focus_clips.get(variant) and name != self.focus_clips[variant]:
            raise ValueError('Clip is outside the selected pilot sample')
        dataset = self.dataset(task).resolve()
        clip = (dataset / name).resolve()
        if clip.parent != dataset or not (clip / 'all_annotations').is_dir():
            raise ValueError('Clip is not in the selected prepared dataset')
        return clip

    def clips(self, task=TASK):
        self.dataset(task)
        dataset = self.dataset(task)
        cached = self._index.get(str(dataset))
        if cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        result = []
        if not dataset.is_dir():
            return result
        manifest = PreparedManifest(self.root)
        variant = 'stabilized' if task.startswith('stabilized') else 'native'
        for entry in manifest.entries(variant):
            if self.focus_clips.get(variant) and entry['clip_id'] != self.focus_clips[variant]:
                continue
            folder = dataset / entry['clip_id']
            try:
                clip = self.clip(folder.name, task)
                meta = read_json(clip / 'all_annotations/meta.json')
                if (meta.get('modality') != 'RGB'
                        or meta.get('annotation_status') not in ('completed', 'completed_minimal')
                        or (clip / 'all_annotations/rejected.json').exists()):
                    continue
                result.append({'id': clip.name, 'status': meta['annotation_status'],
                               'duration_s': meta.get('duration_s'),
                               'has_transcript': bool((meta.get('scene_description') or {}).get('text'))})
            except (OSError, ValueError, TypeError):
                continue
        result.sort(key=lambda row: (not row['has_transcript'], row['id']))
        self._index[str(dataset)] = (time.monotonic(), result)
        return result

    def template(self, task=TASK):
        self.dataset(task)
        prompt = yaml.safe_load((PROMPTS / f'{task}.yaml').read_text())
        return {'task': task, 'title': TASKS[task], 'prompt': prompt,
                'live_preview': True, 'production_request': True, 'focused': bool(self.focus_clips),
                'settings': self.settings,
                'tasks': [{'id': key, 'title': title} for key, title in TASKS.items()]}

    def preview(self, name, task=TASK, category=''):
        self.dataset(task)
        clip = self.clip(name, task)
        available = []
        if 'medical_spatiotemporal' in task:
            available, _ = categories(clip, 'stabilized' if task.startswith('stabilized') else 'native')
            if self.focus_category:
                if self.focus_category not in available or (category and category != self.focus_category):
                    raise ValueError('Category is outside the selected pilot or unavailable')
                available = [self.focus_category]
            category = category or next(iter(available), '')
        prepared = prepare(clip, PROMPTS / f'{task}.yaml', self.settings, category or None)
        request = request_payload(prepared, '<uploaded_video_uri>')
        result_path = (clip / 'GT/medical_spatiotemporal_vqa' / f'{category}.json' if category
                       else clip / 'GT/general_scene_description.json')
        result = None
        if result_path.is_file() and not result_path.is_symlink() and result_path.resolve().is_relative_to(clip):
            saved = read_json(result_path)
            generation = saved.get('generation', {})
            previous_inputs = copy.deepcopy(generation.get('inputs', {}))
            current_inputs = copy.deepcopy(prepared['inputs'])
            for inputs in (previous_inputs, current_inputs):
                inputs.get('implementation', {}).pop('preview.py', None)
            result = {'response': saved.get('response'), 'current': is_current(result_path, prepared),
                      'request_unchanged': previous_inputs == current_inputs,
                      'created_at': generation.get('created_at'),
                      'model': generation.get('inputs', {}).get('settings', {}).get('model'),
                      'usage': generation.get('usage', {}),
                      'checks': response_checks(saved.get('response'), prepared),
                      'review': review_response(saved.get('response') or {})}

        return dict(clip=name, task=task, result=result, category=category, categories=available, evidence_frames=prepared.get('evidence_frames', []), version=prepared['prompt']['version'],
                    system_prompt=request['messages'][0]['content'],
                    user_prompt=request['messages'][1]['content'][1]['text'],
                    schema=request['response_format']['json_schema']['schema'],
                    settings=prepared['settings'], request=request, production_request=True,
                    human_evidence={cue: {'points': values['points'], 'spans': values['spans']}
                                    for cue, values in prepared.get('evidence', {}).items()},
                    video_url='/video?task=' + task + '&id=' + name,
                    inputs=prepared['inputs'], context_exclusions=prepared.get('context_exclusions', {}),
                    note='Built by generate.py prepare() and request_payload(). Media URIs are placeholders. Medical evidence frames are attached during generation; no upload or API call is made here.')

    def evidence_frame(self, name, task, category, index):
        if 'medical_spatiotemporal' not in task:
            raise ValueError('Evidence frames apply only to medical tasks')
        clip = self.clip(name, task)
        if self.focus_category and category != self.focus_category:
            raise ValueError('Category is outside the selected pilot')
        prepared = prepare(clip, PROMPTS / f'{task}.yaml', self.settings, category)
        frames = [frame for frame in prepared['evidence_frames'] if frame['frame_index'] == index]
        if not frames:
            raise ValueError('Frame is not a supplied human evidence frame')
        parts = evidence_images(dict(prepared, evidence_frames=frames))
        return base64.b64decode(parts[1]['image_url']['url'].split(',', 1)[1], validate=True)

    def video(self, name, task=TASK):
        clip = self.clip(name, task)
        video = (clip / f'{clip.name}.mp4').resolve()
        if video.parent != clip or not video.is_file():
            raise ValueError('Prepared video is missing')
        return video


def handler_for(review):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_headers(self, status, content_type, length):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; frame-ancestors 'none'")

        def json(self, payload, status=200):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_headers(status, 'application/json; charset=utf-8', len(data))
            self.end_headers()
            self.wfile.write(data)

        def stream_video(self, name, task):
            path = review.video(name, task)
            with path.open('rb') as stream:
                size = os.fstat(stream.fileno()).st_size
                start, end, status = 0, size - 1, 200
                value = self.headers.get('Range')
                if value:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
                    if not match or not any(match.groups()):
                        self.send_error(416)
                        return
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), size - 1) if right else size - 1
                    else:
                        start = max(0, size - int(right))
                    if start >= size or end < start:
                        self.send_response(416)
                        self.send_header('Content-Range', f'bytes */{size}')
                        self.end_headers()
                        return
                    status = 206
                length = end - start + 1
                self.send_headers(status, 'video/mp4', length)
                self.send_header('Accept-Ranges', 'bytes')
                if status == 206:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
                self.end_headers()
                stream.seek(start)
                while length:
                    data = stream.read(min(length, 1024 * 1024))
                    if not data:
                        break
                    self.wfile.write(data)
                    length -= len(data)

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                name = query.get('id', [''])[0]
                task = query.get('task', [TASK])[0]
                if url.path == '/api/template':
                    self.json(review.template(task))
                elif url.path == '/api/clips':
                    self.json({'clips': review.clips(task)})
                elif url.path == '/api/clip':
                    self.json(review.preview(name, task, query.get('category', [''])[0]))
                elif url.path == '/evidence-frame':
                    data = review.evidence_frame(name, task, query.get('category', [''])[0], int(query.get('frame', ['-1'])[0]))
                    self.send_headers(200, 'image/png', len(data))
                    self.end_headers()
                    self.wfile.write(data)
                elif url.path == '/video':
                    self.stream_video(name, task)
                elif url.path in ('/', '/app.js', '/style.css'):
                    file = ASSETS / {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}[url.path]
                    kind = {'/': 'text/html', '/app.js': 'text/javascript', '/style.css': 'text/css'}[url.path]
                    data = file.read_bytes()
                    self.send_headers(200, kind + '; charset=utf-8', len(data))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, OSError, TypeError) as error:
                self.json({'error': str(error)}, 400)

    return Handler


def main():
    load_dotenv(REPO / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=(os.getenv('AIRLAB_DATASET_ROOT') or os.getenv('TRIAGE_DATASET_ROOT')))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--native-clip', help='Show only this native pilot clip')
    parser.add_argument('--stabilized-clip', help='Show only this stabilized pilot clip')
    parser.add_argument('--category', help='Show only this medical pilot category')
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error('Set TRIAGE_DATASET_ROOT or pass --dataset-root')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if bool(args.native_clip) != bool(args.stabilized_clip):
        parser.error('Provide both --native-clip and --stabilized-clip for a paired pilot')
    focus = {'native': args.native_clip, 'stabilized': args.stabilized_clip} if args.native_clip else None
    review = Review(args.dataset_root.expanduser(), generation_settings(os.getenv('GEMINI_VIDEO_FPS', 'source')), focus, args.category)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(review))
    print(f'CoT input preview: http://127.0.0.1:{args.port} (read-only; no API calls)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
