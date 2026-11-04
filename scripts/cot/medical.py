"""Assemble category inputs and check generated spatial/temporal references."""
import copy
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.shared.data import category_prompt_values, extract_answers, label_keys
from scripts.train.shared.grounding import frame_index, timestamp, valid_point, video_geometry
from evidence_media import frame_times, frame_reference
from source_cache import read_json, read_yaml

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / 'scripts/train/configs/catalogs/RGB/label_catalog.yaml'
TEMPLATES = REPO / 'scripts/train/prompts/answer_only/RGB/medical_vqa.yaml'


def reference_transcript(text):
    """Represent an absent reference consistently without inventing narration."""
    if text is not None and not isinstance(text, str):
        raise ValueError('Reference transcript must be text')
    return text if text and text.strip() else 'Not provided.'


def categories(clip, variant):
    labels = read_json(clip / 'all_annotations/labels.json')
    answers, excluded = extract_answers(labels, read_yaml(CATALOG), variant)
    return list(answers), excluded


def model_point(point):
    return {key: point[key] for key in ('point', 'time', 'slot') if key in point}


def annotated_cues(annotated, answer):
    return {name: value for group in annotated.values() if isinstance(group, dict)
            for name, value in group.get('cues', {}).items() if name in answer}


def formatted_spans(cue, duration, name):
    result = []
    for span in cue.get('time_spans', []):
        start, end = span['start_s'], span['end_s']
        if not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)) or not 0 <= start < end <= duration + .001:
            raise ValueError(f'Invalid time span: {name}')
        result.append({'start': timestamp(start), 'end': timestamp(end)})
    return result


def scene_medical_context(clip, meta, variant):
    """Supply answered category labels as caption consistency context, without evidence locations."""
    labels = read_json(clip / 'all_annotations/labels.json')
    catalog = read_yaml(CATALOG)
    answers, excluded = extract_answers(labels, catalog, variant)
    if not answers:
        raise ValueError('No valid human medical answers available for scene consistency')
    context = {}
    for category, answer in answers.items():
        definition = catalog['categories'][category]
        cues = annotated_cues(labels['categories'][definition['name']], answer)
        records = []
        for group in definition['subcategories'].values():
            for label in group['cues']:
                name = label['name']
                record = {'label': name, 'answer': answer[name]}
                if answer[name]:
                    spans = formatted_spans(cues[name], meta['duration_s'], name)
                    if spans:
                        record['spans'] = spans
                records.append(record)
        context[category] = {'category': definition['name'], 'labels': records}
    return context, excluded


def medical_inputs(clip, meta, viewpoint, category, variant, base_schema):
    labels = read_json(clip / 'all_annotations/labels.json')
    catalog, templates = read_yaml(CATALOG), read_yaml(TEMPLATES)
    answers, excluded = extract_answers(labels, catalog, variant)
    if category not in answers:
        raise ValueError(f'Category unavailable: {category}: {excluded.get(category, "excluded for this view")}')
    answer, definition = answers[category], catalog['categories'][category]
    annotated = labels['categories'][definition['name']]
    cues = annotated_cues(annotated, answer)
    fps, count, width, height = video_geometry(meta)
    if not meta['frame_convention'].get('coordinates', '').startswith('absolute pixels'):
        raise ValueError('Evidence coordinates must declare absolute pixels')
    video = clip / (clip.name + '.mp4')
    times = frame_times(video, video.stat().st_mtime_ns, video.stat().st_size)
    if len(times) != count:
        raise ValueError('Decoded frame count differs from annotation metadata')
    evidence, requirements, frames = {}, {}, {}

    def point_record(seed):
        index = frame_index(seed['frame'], meta)
        xy = seed['point']
        if not valid_point(xy, width, height):
            raise ValueError('Evidence point is outside its video frame')
        value = {'point': [int(math.floor(1000*xy[0]/width + .5)), int(math.floor(1000*xy[1]/height + .5))],
                 'time': timestamp(times[index]), 'frame_index': index}
        slot = seed.get('slot', seed.get('side'))
        if slot is not None:
            value['slot'] = slot
        return value

    for name, cue in cues.items():
        requirements[name] = {k: cue.get(k) for k in
                             ('answered', 'answer', 'temporal', 'evidence_min', 'evidence_slots', 'evidence_side_options')}
        item = {'points': [], 'tracks': [], 'spans': []}
        evidence[name] = item
        if not answer[name]:
            continue
        minimum = cue.get('evidence_min') or 0
        if type(minimum) is not int or minimum < 0:
            raise ValueError(f'Invalid evidence minimum: {name}')
        if minimum > 0:
            for seed in cue.get('evidence_points', []):
                point = point_record(seed)
                if point not in item['points']:
                    item['points'].append(point)
                frames.setdefault(point['frame_index'], set()).add(name)
        if len(item['points']) < minimum:
            raise ValueError(f'Mandatory human evidence missing: {name}')
        slots = cue.get('evidence_slots') or []
        if minimum > 0 and not set(slots) <= {p.get('slot') for p in item['points']}:
            raise ValueError(f'Mandatory evidence slots missing: {name}')
        if cue.get('temporal'):
            item['spans'] = formatted_spans(cue, meta['duration_s'], name)
        if cue.get('temporal') and not item['spans']:
            raise ValueError(f'Temporal cue lacks human spans: {name}')
    tracks_path = clip / 'all_annotations/points.json'
    if tracks_path.exists():
        for row in read_json(tracks_path).get('frames', []):
            for track in row.get('parts', []):
                part = track.get('part', '')
                name = part.removeprefix('evidence:').split('#')[0]
                if not part.startswith('evidence:') or name not in answer or not answer[name]:
                    continue
                if not (requirements[name].get('evidence_min') or 0) > 0:
                    continue
                if track.get('occluded') is not False or track.get('point') is None:
                    continue
                if not valid_point(track['point'], width, height):
                    continue
                value = point_record(dict(track, frame=row['frame']))
                value['by'] = track.get('by')
                evidence[name]['tracks'].append(value)
    values = category_prompt_values(definition, templates, viewpoint)
    question = values['viewpoint_note'] + '\n\n' + templates['task_templates']['single_turn']
    schema = copy.deepcopy(base_schema)
    schema['properties']['answer'] = {
        'type': 'object', 'properties': {key: {'type': 'integer', 'enum': [0, 1]} for key in label_keys(definition)},
        'required': label_keys(definition), 'additionalProperties': False}
    manifest = [{'frame_index': index, 'time': timestamp(times[index]), 'cues': sorted(names)}
                for index, names in sorted(frames.items())]
    model_requirements = {name: {key: value for key, value in requirement.items() if key != 'answered'}
                          for name, requirement in requirements.items()}
    model_evidence = {
        name: {'points': [model_point(point) for point in item['points']],
               'tracks': [model_point(point) for point in item['tracks']],
               'spans': item['spans']}
        for name, item in evidence.items()
    }
    payload = {
        'category_name': definition['name'], 'deployment_prompt': question,
        'cue_definitions': values['definitions'],
        'selection_rules': values['selection_rules'] + values['assessment_instruction'],
        'category_annotations_json': json.dumps(model_requirements, ensure_ascii=False),
        'evidence_annotations_json': json.dumps(model_evidence, ensure_ascii=False),
        'evidence_frame_manifest_json': json.dumps([frame_reference(frame) for frame in manifest], ensure_ascii=False),
        'category_transcription': reference_transcript(
            ((annotated.get('narration') or {}).get('transcript') or {}).get('text')),
        'deployment_answer': json.dumps(answer), 'response_schema_json': json.dumps(schema)}
    return payload, schema, {'expected_answer': answer, 'evidence': evidence,
                            'requirements': requirements, 'evidence_frames': manifest}


def validate_medical(response, prepared):
    if response['answer'] != prepared['expected_answer']:
        raise ValueError('Generated answer differs from the human answer')
    text = response['reasoning_process']
    fragments = re.findall(r'<evidence>.*?</evidence>', text, flags=re.S)
    if text.count('<evidence>') != len(fragments) or text.count('</evidence>') != len(fragments):
        raise ValueError('Malformed evidence markup')
    cited_points, cited_spans = {}, {}
    for fragment in fragments:
        try:
            node = ET.fromstring(fragment)
        except ET.ParseError as error:
            raise ValueError('Malformed evidence markup') from error
        tags = [child.tag for child in node]
        if len(tags) != len(set(tags)) or node.attrib or any(list(c) or c.attrib for c in node):
            raise ValueError('Duplicate, nested, or attributed evidence tags')
        name = node.findtext('cue')
        if name not in prepared['evidence'] or not prepared['expected_answer'][name]:
            raise ValueError('Evidence refers to an unknown or unselected cue')
        available = prepared['evidence'][name]
        requirement = prepared['requirements'][name]
        if 'point' in tags:
            if not (requirement.get('evidence_min') or 0) > 0:
                raise ValueError('Spatial evidence is not required for this cue')
            if set(tags) not in ({'cue', 'point', 'time'}, {'cue', 'point', 'time', 'slot'}):
                raise ValueError('Invalid point tags')
            point = json.loads(node.findtext('point'))
            if not isinstance(point, list) or len(point) != 2 or any(type(v) is not int or not 0 <= v <= 1000 for v in point):
                raise ValueError('Evidence coordinates must be two integers in [0,1000]')
            time, slot = node.findtext('time'), node.findtext('slot')
            def matches(p):
                return p['point'] == point and p['time'] == time and p.get('slot') == slot
            if not any(matches(p) for p in available['points']):
                raise ValueError('Point, slot, or timestamp does not match supplied human evidence')
            if any(matches(p) for p in available['points']):
                cited_points.setdefault(name, set()).add((tuple(point), time, slot))
        elif set(tags) == {'cue', 'span'}:
            if not requirement.get('temporal'):
                raise ValueError('Temporal spans are not required for this cue')
            value = node.findtext('span')
            if value not in {p['start']+'–'+p['end'] for p in available['spans']}:
                raise ValueError('Span does not match supplied evidence')
            cited_spans.setdefault(name, set()).add(value)
        else:
            raise ValueError('Invalid evidence tags')
    for name, selected in prepared['expected_answer'].items():
        if not selected:
            continue
        requirement = prepared['requirements'][name]
        points = cited_points.get(name, set())
        if len(points) < (requirement.get('evidence_min') or 0):
            raise ValueError(f'Mandatory human evidence not cited: {name}')
        if (requirement.get('evidence_min') or 0) > 0 and not set(requirement.get('evidence_slots') or []) <= {p[2] for p in points}:
            raise ValueError(f'Mandatory slot not cited: {name}')
        if requirement.get('temporal'):
            expected = {p['start']+'–'+p['end'] for p in prepared['evidence'][name]['spans']}
            if not expected <= cited_spans.get(name, set()):
                raise ValueError(f'Mandatory span not cited: {name}')
