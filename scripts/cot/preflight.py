"""Check all requested clip/category inputs before any paid generation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from source_cache import clip_sources


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class PreflightCache:
    """Reuse completed checks; hash JSON contents and track the MP4 file identity.

    A cache miss still runs the full video hash, frame, and annotation checks.
    Generation independently checks sources again before each request/publication.
    """

    def __init__(self, directory, configuration, entries):
        self.directory = Path(directory)
        self.configuration = configuration
        self.entries = entries

    def _path(self, clip):
        return self.directory / (_fingerprint(str(clip.absolute())) + '.json')

    def _key(self, clip, variant):
        annotations = clip / 'all_annotations'
        gt = clip / 'GT'
        medical = gt / 'medical_spatiotemporal_vqa'
        for folder in (clip, annotations, gt, medical):
            if folder.is_symlink():
                raise ValueError('Symlink cannot use the preflight cache')
        video = clip / (clip.name + '.mp4')
        if video.is_symlink():
            raise ValueError('Symlink cannot use the preflight cache')
        stat = video.stat()
        video_identity = [stat.st_dev, stat.st_ino, stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns]
        paths = [clip / 'provenance.json', clip / (clip.name + '.json'),
                 *annotations.glob('*.json'), *medical.glob('*.json')]
        scene = gt / 'general_scene_description.json'
        if scene.exists() or scene.is_symlink():
            paths.append(scene)
        hashes = {}
        for path in sorted(paths):
            if path.is_symlink():
                raise ValueError('Symlink cannot use the preflight cache')
            hashes[str(path.relative_to(clip))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return _fingerprint({'configuration': self.configuration, 'clip': str(clip.absolute()),
                             'entry': self.entries[variant][clip.name],
                             'video': video_identity, 'json': hashes})

    def get(self, clip, variant):
        key = None
        try:
            key = self._key(clip, variant)
            saved = json.loads(self._path(clip).read_text())
            payload = saved['payload']
            if (saved.get('version') == 1 and payload['key'] == key
                    and saved['checksum'] == _fingerprint(payload)):
                return key, payload['reports']
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return key, None

    def put(self, clip, variant, key, reports):
        if key is None or not reports or any(row['status'] == 'blocked' for row in reports):
            return False
        temporary = None
        try:
            if self._key(clip, variant) != key:
                return False
            payload = {'key': key, 'reports': reports}
            document = {'version': 1, 'payload': payload, 'checksum': _fingerprint(payload)}
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', dir=self.directory, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(document, stream)
            temporary.replace(self._path(clip))
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def build_plan(tasks, selected_by_variant, args, settings, *, engine=None):
    if engine is None:
        import generate as engine
    manifest = engine.PreparedManifest(args.dataset_root)
    entries = {variant: {row['clip_id']: row for row in manifest.entries(variant)}
               for variant in selected_by_variant}
    folder = Path(engine.__file__).parent
    dependencies = [*folder.glob('*.py'), engine.CATALOG, engine.TEMPLATES,
                    engine.REPO / 'scripts/preprocess/prepared_manifest.py',
                    engine.REPO / 'scripts/train/shared/data.py',
                    engine.REPO / 'scripts/train/shared/grounding.py',
                    *(folder / 'yaml/RGB' / f'{task}.yaml' for task in tasks)]
    configuration = _fingerprint({
        'tasks': tasks, 'category': args.category, 'settings': settings,
        'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies},
        'libraries': {name: engine.version(name) for name in ('litellm', 'google-genai', 'jsonschema', 'PyYAML')},
    })
    cache = PreflightCache(engine.REPO / '.local-reports/cot/preflight-cache', configuration, entries)
    plan = {'created_at': datetime.now(timezone.utc).isoformat(), 'tasks': {}, 'errors': [],
            'validation_cache': {'reused_clips': 0, 'checked_clips': 0}}
    for task in tasks:
        plan['tasks'][task] = {'jobs': {}, 'counts': {'current': 0, 'ready': 0, 'blocked': 0}}
    work = [(variant, clip) for variant, clips in selected_by_variant.items() for clip in clips]

    def check_clip(variant, clip):
        cache_key, cached = cache.get(clip, variant)
        if cached is not None:
            return cached, True
        reports = []
        with clip_sources():
            for task in tasks:
                if engine.TASKS[task] != variant:
                    continue
                prompt = Path(engine.__file__).parent / 'yaml' / 'RGB' / f'{task}.yaml'
                names = [None]
                if 'medical_spatiotemporal' in task:
                    try:
                        names, excluded = engine.categories(clip, variant)
                        if args.category:
                            names, excluded = [args.category], {}
                        for category, error in excluded.items():
                            reports.append({'task': task, 'clip': str(clip), 'category': category,
                                            'status': 'blocked', 'error': error})
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        reports.append({'task': task, 'clip': str(clip), 'category': None,
                                        'status': 'blocked', 'error': str(error)})
                        continue
                for category in names:
                    report = {'task': task, 'clip': str(clip), 'category': category}
                    try:
                        prepared = engine.prepare(clip, prompt, settings, category)
                        output = (clip / 'GT/medical_spatiotemporal_vqa' / f'{category}.json' if category
                                  else clip / 'GT/general_scene_description.json')
                        if output.parent.is_symlink() or output.is_symlink() or not output.resolve().is_relative_to(clip.resolve()):
                            raise ValueError('Output path escapes the clip folder')
                        report.update(status='current' if engine.is_current(output, prepared) else 'ready',
                                      signature=prepared['signature'])
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        report.update(status='blocked', error=str(error))
                    reports.append(report)
        cache.put(clip, variant, cache_key, reports)
        return reports, False

    workers = engine.effective_workers(args.workers, len(work))
    print(f'Preflight: checking {len(work)} clips across {len(tasks)} tasks | Workers: {workers}',
          file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cot-preflight') as executor:
        futures = [executor.submit(check_clip, variant, clip) for variant, clip in work]
        for finished, future in enumerate(as_completed(futures), 1):
            reports, reused = future.result()
            plan['validation_cache']['reused_clips' if reused else 'checked_clips'] += 1
            for report in reports:
                task_plan = plan['tasks'][report['task']]
                task_plan['counts'][report['status']] += 1
                if report['status'] == 'ready':
                    task_plan['jobs'].setdefault(report['clip'], []).append(
                        {'category': report['category'], 'signature': report['signature']})
                elif report['status'] == 'blocked':
                    plan['errors'].append(report)
            if finished % 25 == 0 or finished == len(work):
                print(f'Preflight: checked {finished}/{len(work)} clips | '
                      f'reused checks: {plan["validation_cache"]["reused_clips"]} | '
                      f'blocked categories/tasks: {len(plan["errors"])}', file=sys.stderr, flush=True)
    manifest.verify_unchanged()
    for task, task_plan in plan['tasks'].items():
        counts = task_plan['counts']
        print(f'Preflight {task}: unchanged {counts["current"]} | '
              f'to generate {counts["ready"]} | blocked {counts["blocked"]}', file=sys.stderr, flush=True)
    return plan
