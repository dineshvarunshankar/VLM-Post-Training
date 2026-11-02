"""Attach validated teacher targets without changing frozen human split membership."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import jsonschema

from scripts.train.shared.data import clip_path, read_json, read_yaml, sha256
from scripts.train.shared.paths import REPO_ROOT

GENERATOR = REPO_ROOT / 'scripts/cot'
CATALOG = REPO_ROOT / 'scripts/train/configs/catalogs/RGB/label_catalog.yaml'
ASSESSMENT = REPO_ROOT / 'scripts/train/prompts/answer_only/RGB/medical_vqa.yaml'
STUDENT_PROMPTS = REPO_ROOT / 'scripts/train/configs/cot/prompts.yaml'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def generator_medical():
    # The generator is also a standalone CLI with sibling imports. Import its
    # existing validator without executing generate.py or touching its settings.
    directory = str(GENERATOR)
    sys.path.insert(0, directory)
    try:
        return importlib.import_module('scripts.cot.medical')
    finally:
        sys.path.remove(directory)


class SourceManifest:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}
        self.paths = {}
        self.stats = {}

    def pin(self, path, *, dataset=True):
        path = Path(path)
        resolved = path.resolve()
        if path.is_symlink() or (dataset and not resolved.is_relative_to(self.root)):
            raise ValueError(f'Path escapes selected dataset root: {path}')
        anchor = self.root if dataset else REPO_ROOT.resolve()
        if not resolved.is_relative_to(anchor):
            raise ValueError(f'Path escapes the declared source namespace: {path}')
        key = ('dataset/' if dataset else 'repository/') + resolved.relative_to(anchor).as_posix()
        if key not in self.files:
            before = resolved.stat()
            with resolved.open('rb') as stream:
                value = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = resolved.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ValueError(f'Source changed while hashing: {path}')
            self.files[key] = value
            self.paths[key] = str(resolved)
            self.stats[str(resolved)] = (after.st_mtime_ns, after.st_size)
        return self.files[key]

    def verify_unchanged(self):
        for name, expected in self.stats.items():
            stat = Path(name).stat()
            if (stat.st_mtime_ns, stat.st_size) != expected:
                raise ValueError(f'Source changed during validation: {name}')


def validate_clip_sources(row, manifest):
    """Pin inference sources independently of generated teacher availability."""
    video = Path(row['video'])
    folder, annotations = video.parent, video.parent / 'all_annotations'
    sources = {'video_sha256':manifest.pin(video),
               'provenance_sha256':manifest.pin(folder / 'provenance.json'),
               'sidecar_sha256':manifest.pin(video.with_suffix('.json')),
               'annotation_sha256':{p.name:manifest.pin(p) for p in sorted(annotations.glob('*.json'))}}
    manifest.pin(folder / 'GT/medical_vqa.json')
    provenance = read_json(folder / 'provenance.json')
    meta = read_json(annotations / 'meta.json')
    if provenance.get('clip') != meta.get('clip') or Path(meta['video_path']).name != video.name:
        raise ValueError('Clip identity differs from writer provenance or selected video')
    if not sources['annotation_sha256'] or provenance.get('annotation_sha256') != sources['annotation_sha256']:
        raise ValueError('Current annotation sha256 differs from writer provenance')
    if provenance.get('sidecar_sha256') != sources['sidecar_sha256']:
        raise ValueError('Sidecar sha256 differs from writer provenance')
    if provenance.get('output_sha256') != sources['video_sha256']:
        raise ValueError('Current video sha256 differs from writer provenance')
    if row.get('video_sha256') != sources['video_sha256']:
        raise ValueError('Video sha256 differs from frozen human row')
    return sources


def validate_target(path, row, catalog, config, manifest):
    sources = validate_clip_sources(row, manifest)
    if not path.is_file():
        raise ValueError(f'Missing generated target: {path}; regenerate against the selected sources')
    manifest.pin(path)
    saved = read_json(path)
    generation, response = saved['generation'], saved['response']
    inputs = generation['inputs']
    if generation['signature'] != fingerprint(inputs):
        raise ValueError('Invalid generation source signature')
    category = row['reference']['category'] if row['task'] == 'medical_vqa' else None
    variant = row['reference']['variant']
    kind = 'medical_spatiotemporal_vqa' if category else 'general_scene_description'
    task = variant + '_' + kind
    if generation['task'] != task or generation.get('category') != category:
        raise ValueError('Generated target task identity differs from selected row')
    folder = Path(row['video']).parent
    from scripts.train.shared import cot_source_compat as compatibility
    implementation = inputs.get('implementation')
    current_base = {p.name:manifest.pin(p,dataset=False) for p in GENERATOR.glob('*.py')}
    modern = compatibility.known_implementation(implementation)
    if not modern and implementation != current_base:
        raise ValueError(f'Unsupported generator implementation fingerprint {fingerprint(implementation)}; unknown profiles require source compatibility review')
    selected = config['cot']['accepted_implementation_fingerprints']
    if modern and selected and fingerprint(implementation) not in selected:
        raise ValueError('Generator implementation is outside configured audited profile selection')
    prompt_path = (compatibility.TEACHER_PROMPTS if modern else GENERATOR / 'yaml/RGB') / (task + '.yaml')
    manifest.pin(compatibility.PROFILE_PATH,dataset=False)
    manifest.pin(Path(compatibility.__file__),dataset=False)
    expected = {
        'video_sha256': sources['video_sha256'],
        'provenance_sha256': sources['provenance_sha256'],
        'catalog_sha256': manifest.pin(CATALOG, dataset=False),
        'prompt_sha256': manifest.pin(prompt_path, dataset=False),
    }
    if category:
        expected['assessment_prompts_sha256'] = manifest.pin(ASSESSMENT, dataset=False)
        if inputs.get('category') != category:
            raise ValueError('Medical category differs from generation inputs')
    for name, value in expected.items():
        if inputs.get(name) != value:
            raise ValueError(f'Generator/prompt version differs from this checkout: {name}: {path}' if name in ('prompt_sha256','assessment_prompts_sha256') else f'Stale {name}: {path}')
    actual = sources['annotation_sha256']
    if inputs.get('annotation_sha256') != actual:
        raise ValueError(f'Stale annotation_sha256: {path}')
    medical = generator_medical()
    prompt = read_yaml(prompt_path)
    values, schema, prepared, duration_fields = compatibility.assemble_inputs(folder,variant,category,prompt,modern=modern,medical=medical)
    if inputs.get('assembled_input_sha256') != fingerprint(values):
        raise ValueError('Stale assembled_input_sha256: reconstructed teacher inputs differ from generation')
    for name in ('resolved_video_duration_s','duration_source'):
        if inputs.get(name) != duration_fields.get(name):
            raise ValueError(f'Stale {name}: prepared video duration differs from generation')
    if modern:
        rebuilt = dict(inputs, **expected, annotation_sha256=actual,
                       assembled_input_sha256=fingerprint(values),implementation=compatibility.current_implementation())
        for name in ('resolved_video_duration_s','duration_source','video_transport'):
            rebuilt.pop(name,None)
        rebuilt.update(duration_fields)
        if inputs['settings'].get('api_base') and variant=='native':
            rebuilt['video_transport'] = dict(compatibility.TRANSPORT_PROFILE)
        profiles = read_json(compatibility.PROFILE_PATH)['profiles']
        libraries = next(iter(profiles.values()))['libraries']
        rebuilt['libraries'] = libraries
        if inputs != rebuilt and not compatibility.compatible_inputs(inputs,rebuilt):
            raise ValueError('Generated source inputs differ from the audited compatibility contract')
    text = response.get('reasoning_process' if category else 'answer')
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Missing nonempty reasoning' if category else 'Missing nonempty caption')
    frames = []
    if category:
        if response.get('answer') != json.loads(row['answer']):
            raise ValueError('Generated answer differs from frozen human GT')
        if prepared['expected_answer'] != json.loads(row['answer']):
            raise ValueError('Current human annotations differ from frozen human GT')
        medical.validate_medical(response,prepared)
        frames = sorted({f['frame_index'] for f in prepared['evidence_frames']})
    jsonschema.validate(response, schema)
    return dict(reasoning=text if category else None,
                answer=row['answer'] if category else text,
                required_frame_indices=frames,
                target_identity={'path':str(path.resolve()), 'sha256':manifest.pin(path),
                                 'source_signature':generation['signature'],
                                 'implementation_fingerprint':fingerprint(implementation)},
                teacher_available=True)


def camera_rows(rows, catalog, config):
    """Reuse the existing direct camera task, with all other auxiliaries disabled."""
    if not rows or not config['mixture']['camera_view']['enabled']:
        return [], {}
    from scripts.train.shared.auxiliary import build_mixture
    policy = deepcopy(config)
    for name in ('medical_vqa', 'counting', 'subject_localization', 'label_questions'):
        policy['mixture'][name]['enabled'] = False
    policy['mixture']['camera_view'].update(sources=['native'], questions_per_video_per_epoch=1)
    native = [r for r in rows if r['reference']['variant'] == 'native']
    if not native:
        return [], {}
    pool, _, report = build_mixture(native, catalog, read_yaml(ASSESSMENT), policy)
    if report['excluded_candidates']:
        raise ValueError(f'Camera task construction failed: {report["excluded_candidates"]}')
    for row in pool:
        row.update(reasoning=None, teacher_available=True, target_identity={'human_annotations':row['annotation_hashes']})
    return pool, report.get('task_prompt_hashes', {})


def build_cot_splits(splits, *, media_root, catalog, config):
    """Return rows and an actionable report; callers must gate training on release_ready.

    Missing held-out teachers never remove medical scoring examples. No training
    substitution or reduced pilot is inferred from incomplete coverage.
    """
    if catalog != read_yaml(CATALOG):
        raise ValueError('CoT requires the supported generator catalog; catalog differs')
    manifest = SourceManifest(media_root)
    prompts = read_yaml(STUDENT_PROMPTS)
    manifest.pin(STUDENT_PROMPTS, dataset=False)
    result, issues, coverage, teacher_loss = {}, [], {}, {}
    owners, seen, checked_sources = {}, set(), set()
    auxiliary, prompt_hashes = {}, {}
    for split, source_rows in splits.items():
        rows, videos = [], {}
        counts = defaultdict(lambda: {'expected':0, 'accepted':0})
        for original in source_rows:
            ref = original['reference']
            key = (ref['variant'], ref['clip'], ref['category'])
            if key in seen:
                raise ValueError(f'Duplicate target or leakage across split membership: {key}')
            seen.add(key)
            folder = clip_path(media_root, ref['variant'], ref['clip'])
            video = folder / (ref['clip'] + '.mp4')
            if Path(original['video']).resolve() != video.resolve():
                raise ValueError('Video path differs from selected split identity')
            from scripts.train.shared.data import recording_group
            manifest.pin(folder / 'GT/medical_vqa.json')
            manifest.pin(folder / 'all_annotations/meta.json')
            meta = read_json(folder / 'all_annotations/meta.json')
            if str(video) not in checked_sources:
                checked_sources.add(str(video))
                try:
                    validate_clip_sources(original, manifest)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issues.append(dict(split=split, task='source', reference=ref,
                                       path=str(video), reason=str(error), blocks_release=True))
            group = recording_group(meta)
            if group in owners and owners[group] != split:
                raise ValueError(f'Source recording appears across splits: {group}')
            owners[group] = split
            row = dict(original, task='medical_vqa', reasoning=None, teacher_available=False,
                       required_frame_indices=[], target_identity=None)
            row['system'] = original['system'] + '\n' + prompts['medical_instruction']
            rows.append(row)
            videos.setdefault(row['video'], row)
            counts['medical_vqa']['expected'] += 1
            path = folder / 'GT/medical_spatiotemporal_vqa' / (ref['category'] + '.json')
            try:
                row.update(validate_target(path, row, catalog, config, manifest))
                counts['medical_vqa']['accepted'] += 1
            except (OSError, ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
                issues.append(dict(split=split, task='medical_vqa', reference=ref, path=str(path), reason=str(error), blocks_release=split=='train'))
        teacher_loss[split] = dict(expected=len(source_rows), covered=counts['medical_vqa']['accepted'])
        direct = []
        for base in videos.values():
            row = dict(base, task='scene_description', system=prompts['caption_system'],
                       question=prompts['caption_question'], reasoning=None, answer='', teacher_available=False,
                       target_identity=None, required_frame_indices=[])
            row['reference'] = dict(base['reference'], category=None, task='scene_description')
            counts['scene_description']['expected'] += 1
            path = Path(base['video']).parent / 'GT/general_scene_description.json'
            try:
                row.update(validate_target(path, row, catalog, config, manifest))
                counts['scene_description']['accepted'] += 1
            except (OSError, ValueError, KeyError, TypeError, jsonschema.ValidationError) as error:
                issues.append(dict(split=split, task='scene_description', reference=row['reference'], path=str(path), reason=str(error), blocks_release=split=='train'))
            direct.append(row)
        try:
            camera, hashes = camera_rows(source_rows, catalog, config)
            prompt_hashes.update(hashes)
        except (OSError, ValueError, KeyError, TypeError) as error:
            camera = []
            issues.append(dict(split=split, task='camera_view', reason=str(error), blocks_release=split=='train'))
        direct.extend(camera)
        counts['camera_view'] = dict(expected=len({r['video'] for r in source_rows if r['reference']['variant']=='native'}) if config['mixture']['camera_view']['enabled'] else 0, accepted=len(camera))
        result[split] = rows + direct if split == 'train' else rows
        auxiliary[split] = direct if split != 'train' else []
        coverage[split] = dict(counts)
    for task in prompt_hashes:
        from scripts.train.shared.auxiliary import PROMPT_DIRECTORY
        manifest.pin(PROMPT_DIRECTORY / (task + '.yaml'), dataset=False)
    manifest.verify_unchanged()
    files = dict(sorted(manifest.files.items()))
    report = dict(schema_version=1, release_ready=not any(i['blocks_release'] for i in issues),
                  coverage=coverage, teacher_loss=teacher_loss, issues=issues,
                  manifest={'schema_version':1,'files':files,'sha256':fingerprint(files)},
                  source_paths={'dataset_root':str(manifest.root),'repository_root':str(REPO_ROOT.resolve()),'files':dict(sorted(manifest.paths.items()))},
                  auxiliary_evaluation=auxiliary,
                  sampling_plan={'fixed':list(range(len(result['train']))), 'groups':[]},
                  task_counts=dict(Counter(r['task'] for r in result['train'])))
    return result, report


def require_complete_training(report):
    if not report['release_ready']:
        problems = [i for i in report['issues'] if i['blocks_release']]
        raise ValueError(f'CoT dataset release blocked by {len(problems)} missing or invalid targets; inspect coverage report. First issue: {problems[0]}')



def verify_release_manifest(config, report):
    """A training run must name the exact reviewed immutable source manifest."""
    path = config["cot"].get("release_manifest")
    digest = config["cot"].get("release_manifest_sha256")
    if not path or not digest:
        raise ValueError("CoT training requires a reviewed release_manifest and release_manifest_sha256; run CPU inspection first")
    if sha256(path) != digest:
        raise ValueError("CoT release manifest SHA256 changed")
    if read_json(path) != report["manifest"]:
        raise ValueError("CoT selected sources differ from the pinned release manifest; inspect a new release")



def pin_split_membership(report, split_dir):
    files = report["manifest"]["files"]
    paths = [Path(split_dir)/(name+".jsonl") for name in ("train","validation","test")]
    if (Path(split_dir)/"summary.json").is_file():
        paths.append(Path(split_dir)/"summary.json")
    for path in paths:
        key = "splits/" + path.name
        files[key] = sha256(path)
        report["source_paths"]["files"][key] = str(path.resolve())
    report["manifest"]["sha256"] = fingerprint(files)
