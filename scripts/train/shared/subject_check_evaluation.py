"""Separate binary subject-check metrics and additive training membership."""

from copy import deepcopy
import json

KEYS = ('subject_is_human_or_human_like_manikin', 'subject_is_not_human_or_manikin')


def _positive(text):
    from scripts.train.shared.cot_format import unique_object
    value = json.loads(text, object_pairs_hook=unique_object)
    if (not isinstance(value, dict) or set(value) != set(KEYS)
            or any(type(value[k]) is not int or value[k] not in (0, 1) for k in KEYS)
            or sum(value.values()) != 1):
        raise ValueError('Subject check requires exactly one selected binary key')
    return value[KEYS[0]] == 1


def score_subject_check(records):
    def aggregate(rows):
        positives = negatives = correct = false_positive = true_positive = true_negative = invalid = 0
        for row in rows:
            truth = _positive(row['answer'])
            positives += truth
            negatives += not truth
            try:
                prediction = _positive(row['prediction'])
            except (TypeError, ValueError):
                invalid += 1
                continue
            correct += prediction == truth
            false_positive += prediction and not truth
            true_positive += prediction and truth
            true_negative += not prediction and not truth
        return dict(examples=len(rows), positives=positives, negatives=negatives,
                    accuracy=correct / len(rows) if rows else None,
                    false_positive_rate=false_positive / negatives if negatives else None,
                    positive_recall=true_positive / positives if positives else None,
                    negative_recall=true_negative / negatives if negatives else None,
                    invalid_rate=invalid / len(rows) if rows else None,
                    false_positives=false_positive, true_positives=true_positive, true_negatives=true_negative)
    modalities = sorted({row['reference']['modality'] for row in records})
    return dict(metrics=aggregate(records), modalities={m: aggregate([
        row for row in records if row['reference']['modality'] == m]) for m in modalities}, predictions=records)


def subject_metric_values(report, prefix):
    metrics = {f'{prefix}_subject_check/{key}': value for key, value in report['metrics'].items() if value is not None}
    for modality, values in report['modalities'].items():
        metrics.update({f'{prefix}_subject_check/{modality}/{key}': value for key, value in values.items() if value is not None})
    return metrics


def attach_subject_check(splits, plan, report, subjects, provenance):
    splits = dict(splits)
    start = len(splits['train'])
    rows = splits['train'] + subjects['train']
    # Dataset.from_list takes its top-level columns from the first row.
    # Materialize the union before Arrow conversion to retain image payloads.
    columns = set().union(*(row.keys() for row in rows), {'media_type', 'image', 'video'})
    splits['train'] = [{**dict.fromkeys(columns), **row,
                        'media_type': row.get('media_type') or 'video',
                        'task': row.get('task') or 'medical_vqa'} for row in rows]
    plan = deepcopy(plan) if plan is not None else {'fixed': list(range(start)), 'groups': []}
    plan['fixed'].extend(range(start, len(splits['train'])))
    report = deepcopy(report) if report is not None else {}
    if 'task_counts' in report:
        report['task_counts']['subject_check'] = len(subjects['train'])
    report['sampling_plan'] = plan
    report['subject_check'] = provenance
    report['subject_check_evaluation'] = {name: subjects[name] for name in ('validation', 'test')}
    return splits, plan, report


def pin_subject_check_manifest(report, media_root, split_dir):
    """Bind reviewed CoT release acceptance to additive subject sources."""
    from pathlib import Path
    from scripts.train.shared.subject_check import DATASET
    from scripts.train.shared.cot_data import fingerprint
    provenance = report['subject_check']
    manifest = report['manifest']
    paths = {'dataset_manifest.json': Path(media_root) / '10-other_datasets' / DATASET / 'dataset_manifest.json',
             'summary.json': Path(split_dir) / 'subject_check' / 'summary.json'}
    hashes = {'dataset_manifest.json': provenance['manifest_sha256'], 'summary.json': provenance['summary_sha256']}
    for split, digest in provenance['split_sha256'].items():
        paths[split + '.jsonl'] = Path(split_dir) / 'subject_check' / (split + '.jsonl')
        hashes[split + '.jsonl'] = digest
    for name, path in paths.items():
        key = 'subject_check/' + name
        manifest['files'][key] = hashes[name]
        report['source_paths']['files'][key] = str(path.resolve())
    manifest['sha256'] = fingerprint(manifest['files'])
    manifest['subject_check_prompt_sha256'] = provenance['prompt_sha256']
