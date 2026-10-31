"""Prepare private reward metadata and score answers without exposing GT in prompts."""


import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.train.shared.paths import ANSWER_ONLY_PROMPT_DIR
from scripts.train.shared.spatial_policy import spatial_rule
from scripts.train.shared.data import check_answer, label_keys, read_json, read_yaml, sha256
from scripts.train.shared.grounding import temporal_target, video_geometry, subject_targets, normalized_subject_box, human_evidence, spatial_target, specialize_evidence_schema
from scripts.train.rl.rewards import combined_reward, count_reward, medical_reward, number, parse_prediction, spatial_reward, temporal_reward, subject_reward


PROMPTS = ANSWER_ONLY_PROMPT_DIR / 'RGB'




def evidence_subject_boxes(points, boxes):
    """Attach the designated subject box at each verified human evidence frame."""
    import math
    if not boxes.get('box_convention', '').startswith('bbox_pct is percent of frame, origin top-left'):
        raise ValueError('Subject boxes require the declared percent coordinate convention')
    frames = {}
    for row in boxes['frames']:
        if row['frame'] in frames:
            raise ValueError('Duplicate subject box frame')
        frames[row['frame']] = row.get('bbox_pct')
    result = []
    for point in points:
        box = frames.get(point['frame_index'] + 1)
        if not box:
            raise ValueError('Missing subject box at the human evidence frame')
        bounds = normalized_subject_box(box)
        rounded = [int(math.floor(v + .5)) for v in point['point']]
        if not subject_reward(point['point'], bounds) or not subject_reward(rounded, bounds):
            raise ValueError('Human subject-type evidence lies outside the designated subject box')
        result.append(dict(point, box=bounds))
    return result


class RewardPreparer:
    def __init__(self, catalog, reward_config, prompt_directory=PROMPTS):
        self.catalog = catalog
        self.reward_config = reward_config
        self.prompts = Path(prompt_directory)
        self.cache = {}

    def annotation(self, folder, name):
        key = (str(folder), name)
        if key not in self.cache:
            path = folder / 'all_annotations' / name
            provenance = read_json(folder / 'provenance.json')
            if provenance.get('annotation_sha256', {}).get(name) != sha256(path):
                raise ValueError(f'{name} differs from writer provenance; refresh preprocessing')
            self.cache[key] = read_json(path)
        return self.cache[key]

    def prepare(self, row, verified_geometry=None):
        """Return a copied example with reward_context; geometry comes from decoded media."""
        row = deepcopy(row)
        task = row['task']
        target = json.loads(row['answer'])
        context = dict(task=task, target=target)
        reference = row['reference']
        if task == 'medical_vqa':
            definition = self.catalog['categories'][reference['category']]
            check_answer(target, definition)
            keys = label_keys(definition)
            context['definition'] = definition
            schema = dict(type='object', properties={k:dict(type='integer', enum=[0,1]) for k in keys}, required=keys, additionalProperties=False)
        else:
            allowed = {'subject_localization','camera_view','counting','label_selection','label_selection_grounding','label_selection_temporal','label_selection_spatiotemporal'}
            if task not in allowed:
                raise ValueError(f'Task preparation is not implemented: {task}')
            schema = deepcopy(read_yaml(self.prompts / f'{task}.yaml')['response_schema'])
            label = reference.get('label')
            def specialize(item):
                if isinstance(item, dict):
                    return {(label if k == '{label}' else k):specialize(v) for k,v in item.items()}
                if isinstance(item, list):
                    return [specialize(v) for v in item]
                return label if item == '{label}' else item
            schema = specialize(schema)
            if task == 'subject_localization':
                folder = Path(row['video']).parent
                meta = self.annotation(folder, 'meta.json')
                fps, count, width, height = video_geometry(meta)
                expected = dict(width=width,height=height,n_frames=count,fps=fps)
                if verified_geometry is None or any(verified_geometry.get(k) != v for k,v in expected.items()):
                    raise ValueError('Subject rewards require decoded geometry matching annotation metadata')
                candidates = subject_targets(self.annotation(folder,'counting.json'),
                    self.annotation(folder,'points.json'),self.annotation(folder,'boxes.json'),meta)
                matches = [p for p in candidates if p['timestamp'] == reference['timestamp']]
                if len(matches) != 1 or matches[0]['point'] != target['point']:
                    raise ValueError('Subject target differs from the human seed')
                context['subject_box'] = matches[0]['box']
            if task.startswith('label_selection'):
                context['label'] = label
            if task in ('label_selection_grounding','label_selection_temporal','label_selection_spatiotemporal'):
                folder = Path(row['video']).parent
                meta = self.annotation(folder, 'meta.json')
                labels = self.annotation(folder, 'labels.json')
                cues = [sub['cues'][label] for category in labels['categories'].values() for sub in category.values() if isinstance(sub,dict) and label in sub.get('cues',{})]
                if len(cues) != 1:
                    raise ValueError('Expected exactly one human annotation for the label')
                cue = cues[0]
                if cue.get('answered') is not True or type(cue.get('answer')) is not bool or int(cue['answer']) != target['answer'][label]:
                    raise ValueError('Prepared label differs from human answer')
                positive = cue['answer']
                if 'evidence' in target:
                    if type(cue.get('evidence_min')) is not int or cue['evidence_min'] < 1:
                        raise ValueError('Spatial question requires mandatory evidence')
                    rule = spatial_rule(self.reward_config['spatial'], reference['category'], label)
                    if rule['mode'] == 'disabled':
                        raise ValueError('Spatial evidence is disabled for this label')
                    specialize_evidence_schema(schema, cue)
                    if positive:
                        fps, count, width, height = video_geometry(meta)
                        expected = dict(width=width,height=height,n_frames=count,fps=fps)
                        if verified_geometry is None or any(verified_geometry.get(k) != v for k,v in expected.items()):
                            raise ValueError('Spatial rewards require decoded geometry matching annotation metadata')
                        variant = reference['variant']
                        full = rule.get('full_credit_pixels', self.reward_config['spatial'][f'{variant}_full_credit_pixels'])
                        zero = rule.get('zero_credit_pixels', self.reward_config['spatial'][f'{variant}_zero_credit_pixels'])
                        if full is None or zero is None:
                            raise ValueError(f'{variant} spatial tolerance is not finalized')
                        points = human_evidence(cue, meta)
                        if rule['mode'] == 'subject_box':
                            points = evidence_subject_boxes(points, self.annotation(folder, 'boxes.json'))
                        context['spatial'] = dict(
                            target_mode=rule['mode'],
                            targets=points, width=width, height=height, full_pixels=full, zero_pixels=zero,
                            timestamp_tolerance_seconds=self.reward_config['spatial']['timestamp_tolerance_seconds'],
                        )
                        expected_evidence, expected_frames = spatial_target(cue, meta)
                        if target['evidence'] != expected_evidence:
                            raise ValueError('Prepared spatial evidence differs from shared human targets')
                        if not set(expected_frames) <= set(row.get('required_frame_indices', [])):
                            raise ValueError('Prepared video omits required human evidence frames')
                    elif target['evidence']:
                        raise ValueError('Negative GT must have empty evidence')
                if 'time_spans' in target:
                    if cue.get('temporal') is not True:
                        raise ValueError('Span question requires a mandatory temporal label')
                    if positive:
                        if not number(meta.get('duration_s')) or meta['duration_s'] <= 0:
                            raise ValueError('Temporal rewards require a verified video duration')
                        spans, _ = temporal_target(cue,meta)
                        if target['time_spans'] != spans:
                            raise ValueError('Prepared spans differ from human GT')
                        context['duration'] = meta['duration_s']
                    elif target['time_spans']:
                        raise ValueError('Negative GT must have empty spans')
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(target)
        context['schema'] = schema
        row['answer'] = json.dumps(target,separators=(',',':'))
        row['reward_context'] = context
        return row


def score_example(text, context, config):
    """Return total and raw components; malformed predictions score zero, bad GT raises."""
    schema, target, task = context['schema'], context['target'], context['task']
    Draft202012Validator(schema).validate(target)
    try:
        predicted = parse_prediction(text,schema)
    except (ValueError,TypeError,ValidationError) as error:
        return dict(total=0.0,valid=False,components={},error=str(error))
    components = {}
    if task == 'medical_vqa':
        try:
            check_answer(predicted,context['definition'])
        except (ValueError,TypeError) as error:
            return dict(total=0.0,valid=False,components={},error=str(error))
        total = medical_reward(text,target,context['definition'])
        components['answer'] = total
    elif task == 'subject_localization':
        total = subject_reward(predicted['point'],context['subject_box'])
        components['spatial'] = total
    elif task == 'counting':
        total = count_reward(predicted['count'],target['count'])
        components['count'] = total
    elif task in ('camera_view','label_selection'):
        total = float(predicted == target)
        components['answer'] = total
    elif task in ('label_selection_grounding','label_selection_temporal','label_selection_spatiotemporal'):
        label = context['label']
        pred, gt = predicted['answer'][label], target['answer'][label]
        components['answer'] = float(pred == gt)
        spatial, temporal = None, None
        if gt == 1:
            if 'evidence' in target:
                spatial = spatial_reward(predicted['evidence'],**context['spatial'])
                components['spatial'] = spatial
            if 'time_spans' in target:
                t = config['temporal']
                temporal = temporal_reward(predicted['time_spans'],target['time_spans'],context['duration'],t['start_full_seconds'],t['start_zero_seconds'],t['end_full_seconds'],t['end_zero_seconds'])
                components['temporal'] = temporal
        key = 'answer_and_both' if 'evidence' in target and 'time_spans' in target else 'answer_and_spatial' if 'evidence' in target else 'answer_and_temporal'
        empty = all(not predicted[k] for k in ('evidence','time_spans') if k in target)
        total = combined_reward(pred,gt,spatial,temporal,empty,config[key])
    else:
        raise ValueError(f'Reward dispatch is not implemented: {task}')
    return dict(total=total,valid=True,components=components,error=None)


def make_reward_function(config):
    """Adapt private per-example targets to TRL's batch reward callback."""
    def answer_reward(completions, reward_context, log_metric=None, **kwargs):
        if len(completions) != len(reward_context):
            raise ValueError('Completion and reward metadata batch sizes differ')
        totals, records = [], []
        for completion, context in zip(completions,reward_context):
            if isinstance(completion,str):
                text = completion
            elif (isinstance(completion,list) and len(completion)==1
                  and isinstance(completion[0],dict)
                  and completion[0].get('role')=='assistant'
                  and isinstance(completion[0].get('content'),str)):
                text = completion[0]['content']
            else:
                text = ''
            result = score_example(text,context,config)
            totals.append(result['total'])
            records.append((context['task'],result))
        if log_metric is not None and records:
            log_metric('answers/invalid_rate',sum(not r['valid'] for _,r in records)/len(records))
            for task in sorted({task for task,_ in records}):
                selected = [r for name,r in records if name==task]
                log_metric(f'rewards/{task}/total',sum(r['total'] for r in selected)/len(selected))
                for component in sorted({key for r in selected for key in r['components']}):
                    values = [r['components'][component] for r in selected if component in r['components']]
                    log_metric(f'rewards/{task}/{component}',sum(values)/len(values))
        return totals
    return answer_reward
