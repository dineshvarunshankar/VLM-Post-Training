"""Strict answer validation and category-balanced binary-label F1."""

import json
from collections import defaultdict

from scripts.train.shared.data import check_answer, label_keys


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate answer key: {key}")
        result[key] = value
    return result


def parse_answer(text, definition):
    value = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Answer must be a JSON object")
    check_answer(value, definition)
    return value


def score_answers(rows, catalog):
    if not rows:
        raise ValueError("No validation predictions to score")
    counts = defaultdict(
        lambda: defaultdict(
            lambda: dict(true_positive=0, false_positive=0, false_negative=0, support=0)
        )
    )
    invalid, exact, records = 0, 0, []
    category_examples, category_invalid = defaultdict(int), defaultdict(int)
    for row in rows:
        category = row["reference"]["category"]
        definition = catalog["categories"][category]
        target = parse_answer(row["answer"], definition)
        error = None
        try:
            predicted = parse_answer(row["prediction"], definition)
        except (ValueError, TypeError) as exc:
            error = str(exc)
            invalid += 1
            category_invalid[category] += 1
            # An invalid answer is wrong for every expected label, including negatives.
            predicted = {key: 1 - value for key, value in target.items()}
        category_examples[category] += 1
        exact += int(error is None and predicted == target)
        for label in label_keys(definition):
            gt, pred = target[label], predicted[label]
            item = counts[category][label]
            item["true_positive"] += int(gt == 1 and pred == 1)
            item["false_positive"] += int(gt == 0 and pred == 1)
            item["false_negative"] += int(gt == 1 and pred == 0)
            item["support"] += gt
        records.append({**row, "valid": error is None, "error": error})
    categories = {}
    for category, labels in counts.items():
        active = []
        for item in labels.values():
            denominator = (
                2 * item["true_positive"]
                + item["false_positive"]
                + item["false_negative"]
            )
            item["f1"] = (
                2 * item["true_positive"] / denominator if denominator else None
            )
            if denominator:
                active.append(item["f1"])
        categories[category] = dict(
            f1=sum(active) / len(active) if active else None,
            examples=category_examples[category],
            invalid=category_invalid[category],
            labels=dict(labels),
        )
    scores = [value["f1"] for value in categories.values() if value["f1"] is not None]
    if not scores:
        raise ValueError(
            "Validation has no labels with positive ground truth or predictions; F1 is undefined"
        )
    return {
        "metrics": {
            "macro_f1": sum(scores) / len(scores),
            "invalid_rate": invalid / len(rows),
            "exact_match": exact / len(rows),
            "examples": len(rows),
        },
        "categories": categories,
        "predictions": records,
    }


def score_medical_evaluation(rows, catalog, policy):
    """Keep legacy pooled scoring unless a common two-view evaluation is configured."""
    report = score_answers(rows, catalog)
    if not policy.get('sources'):
        return report
    if policy.get('aggregation') != 'equal_source_macro_f1':
        raise ValueError('Unsupported medical evaluation aggregation')
    sources = policy['sources']
    observed = {row['reference']['variant'] for row in rows}
    if observed != set(sources):
        raise ValueError(f'Evaluation sources {observed} do not match required sources {sources}')
    reports = {}
    for source in sources:
        scored = score_answers([row for row in rows if row['reference']['variant'] == source], catalog)
        reports[source] = {key: scored[key] for key in ('metrics', 'categories')}
    report['medical_sources'] = reports
    report['aggregation'] = policy['aggregation']
    report['metrics']['pooled_macro_f1'] = report['metrics']['macro_f1']
    report['metrics']['macro_f1'] = sum(r['metrics']['macro_f1'] for r in reports.values()) / len(reports)
    return report
