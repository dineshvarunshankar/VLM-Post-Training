#!/usr/bin/env python3
"""Read-only localhost review of CoT prompts and available model inputs."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from dotenv import load_dotenv
import yaml

from generate import DATASET, REPO, TASK, generation_settings, prepare, read_json, request_payload

ASSETS = Path(__file__).parent / 'preview'
PROMPTS = Path(__file__).parent / 'yaml' / 'RGB'
PROMPT = PROMPTS / f'{TASK}.yaml'
TASKS = {
    'native_general_scene_description': 'Native scene description',
    'stabilized_general_scene_description': 'Stabilized scene description',
    'native_medical_spatiotemporal_vqa': 'Native medical spatiotemporal VQA',
    'stabilized_medical_spatiotemporal_vqa': 'Stabilized medical spatiotemporal VQA',
}


class Review:
    def __init__(self, root, settings):
        self.dataset = (root / DATASET).resolve()
        if not self.dataset.is_dir():
            raise ValueError('Prepared native dataset folder is missing')
        self.settings = settings

    def clip(self, name):
        if not name or Path(name).name != name or name.startswith('.'):
            raise ValueError('Invalid clip ID')
        clip = (self.dataset / name).resolve()
        if clip.parent != self.dataset or not (clip / 'all_annotations').is_dir():
            raise ValueError('Clip is not in the prepared native dataset')
        return clip

    def clips(self):
        result = []
        for clip in sorted(self.dataset.iterdir()):
            try:
                clip = self.clip(clip.name)
                meta = read_json(clip / 'all_annotations' / 'meta.json')
                if (meta.get('modality') != 'RGB'
                        or meta.get('annotation_status') not in ('completed', 'completed_minimal')
                        or (clip / 'all_annotations' / 'rejected.json').exists()):
                    continue
                result.append({'id': clip.name, 'status': meta['annotation_status'],
                               'duration_s': meta.get('duration_s')})
            except (OSError, ValueError, TypeError):
                continue
        return result

    def template(self, task=TASK):
        if task not in TASKS:
            raise ValueError('Unknown CoT task')
        prompt = yaml.safe_load((PROMPTS / f'{task}.yaml').read_text())
        return {'task': task, 'title': TASKS[task], 'prompt': prompt,
                'live_preview': task == TASK, 'settings': self.settings,
                'tasks': [{'id': key, 'title': title} for key, title in TASKS.items()]}

    def preview(self, name, task=TASK):
        if task != TASK:
            raise ValueError('Clip-specific inputs are implemented only for native scene description')
        prepared = prepare(self.clip(name), PROMPT, self.settings)
        return {'clip': name, 'task': TASK, 'settings': prepared['settings'],
                'request': request_payload(prepared, '<uploaded_video_uri>')}

    def video(self, name):
        clip = self.clip(name)
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

        def stream_video(self, name):
            path = review.video(name)
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
                    self.json({'clips': review.clips()})
                elif url.path == '/api/clip':
                    self.json(review.preview(name, task))
                elif url.path == '/video':
                    self.stream_video(name)
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
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error('Set TRIAGE_DATASET_ROOT or pass --dataset-root')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    review = Review(args.dataset_root.expanduser(), generation_settings(os.getenv('GEMINI_VIDEO_FPS', 'source')))
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
