"""Answer-only rewards. Grounding targets must be prepared from verified human GT."""

import json
import math
import re

from jsonschema import Draft202012Validator

from scripts.train.shared.data import check_answer
from scripts.train.shared.metrics import _unique_object


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def point(value):
    return isinstance(value, list) and len(value) == 2 and all(
        number(v) and 0 <= v <= 1000 for v in value
    )


def seconds(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{2}:[0-5][0-9](\.[0-9]{1,3})?', value):
        raise ValueError('Timestamp must use MM:SS with optional fractional seconds')
    minutes, rest = value.split(':')
    return 60 * int(minutes) + float(rest)


def falloff(error, full, zero):
    if not all(number(v) for v in (error, full, zero)) or not 0 <= full < zero or error < 0:
        raise ValueError('Invalid tolerance or error')
    return max(0.0, min(1.0, (zero - error) / (zero - full)))


def selected_label_f1(predicted, target):
    if not isinstance(predicted, dict) or predicted.keys() != target.keys():
        return 0.0
    if any(type(v) is not int or v not in (0, 1) for v in target.values()):
        raise ValueError('Target labels must be integer 0 or 1')
    if any(type(v) is not int or v not in (0, 1) for v in predicted.values()):
        return 0.0
    true = {k for k, v in target.items() if v}
    selected = {k for k, v in predicted.items() if v}
    denominator = len(true) + len(selected)
    return 2 * len(true & selected) / denominator if denominator else 1.0


def count_reward(predicted, target):
    if type(target) is not int or target < 0:
        raise ValueError('Target count must be a nonnegative integer')
    if type(predicted) is not int or predicted < 0:
        return 0.0
    return max(0.0, 1 - abs(predicted - target) / max(target, 1))


def subject_reward(predicted, box):
    if not isinstance(box, list) or len(box) != 4 or not all(number(v) for v in box):
        raise ValueError('Subject box must use normalized [xmin, ymin, xmax, ymax]')
    x1, y1, x2, y2 = box
    if max(0, x1) >= min(1000, x2) or max(0, y1) >= min(1000, y2):
        raise ValueError('Subject box must have positive visible area inside the frame')
    return float(point(predicted) and x1 <= predicted[0] <= x2 and y1 <= predicted[1] <= y2)


def maximum_matching(weights):
    """Maximum-weight one-to-one assignment, with unmatched entries scoring zero."""
    if not weights or not weights[0]:
        return 0.0
    rows, columns = len(weights), len(weights[0])
    if any(len(row) != columns for row in weights):
        raise ValueError('Matching matrix must be rectangular')
    if any(not number(v) or not 0 <= v <= 1 for row in weights for v in row):
        raise ValueError('Matching scores must lie in [0, 1]')
    if rows > columns:
        weights = [list(row) for row in zip(*weights)]
        rows, columns = columns, rows
    # Hungarian assignment on negative scores; rows <= columns.
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    assignment, previous = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        assignment[0] = row
        column = 0
        minimum, used = [math.inf] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column] = True
            active_row = assignment[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                cost = -weights[active_row - 1][candidate - 1] - u[active_row] - v[candidate]
                if cost < minimum[candidate]:
                    minimum[candidate], previous[candidate] = cost, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[assignment[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if assignment[column] == 0:
                break
        while column:
            last = previous[column]
            assignment[column] = assignment[last]
            column = last
    return sum(weights[r - 1][c - 1] for c, r in enumerate(assignment) if c and r)


def spatial_reward(predicted, targets, width, height, full_pixels, zero_pixels, *, timestamp_tolerance_seconds=0.5, target_mode="point"):
    """Targets contain only required points for one cue, with explicit human provenance."""
    if not targets or not all(number(v) and v > 0 for v in (width, height)):
        raise ValueError('Required human points and frame dimensions must be supplied')
    if not number(timestamp_tolerance_seconds) or timestamp_tolerance_seconds < 0:
        raise ValueError("Evidence timestamp tolerance must be finite and nonnegative")
    falloff(0, full_pixels, zero_pixels)
    if target_mode not in {'point', 'subject_box'}:
        raise ValueError('Unknown spatial target mode')
    for target in targets:
        if target.get('by') != 'human' or not point(target.get('point')):
            raise ValueError('Evidence rewards require verified human points')
        seconds(target['timestamp'])
        if target_mode == 'subject_box':
            subject_reward(target['point'], target['box'])
    def pair(pred, target):
        if not isinstance(pred, dict) or not point(pred.get('point')):
            return 0.0
        try:
            delta = abs(seconds(pred.get('timestamp')) - seconds(target['timestamp']))
            same_time = delta <= timestamp_tolerance_seconds or math.isclose(
                delta, timestamp_tolerance_seconds, rel_tol=0, abs_tol=1e-9
            )
        except ValueError:
            return 0.0
        if not same_time or pred.get('slot') != target.get('slot'):
            return 0.0
        if target_mode == 'subject_box':
            return subject_reward(pred['point'], target['box'])
        dx = (pred['point'][0] - target['point'][0]) * width / 1000
        dy = (pred['point'][1] - target['point'][1]) * height / 1000
        return falloff(math.hypot(dx, dy), full_pixels, zero_pixels)
    weights = [[pair(p, t) for t in targets] for p in predicted]
    return maximum_matching(weights) / max(len(predicted), len(targets))


def temporal_reward(predicted, targets, duration, start_full=.5, start_zero=1., end_full=1., end_zero=2.):
    if not targets or not number(duration) or duration <= 0:
        raise ValueError('Required GT spans and video duration must be supplied')
    falloff(0, start_full, start_zero)
    falloff(0, end_full, end_zero)
    def span(item):
        start, end = seconds(item['start']), seconds(item['end'])
        if not 0 <= start < end <= duration:
            raise ValueError('Span must be ordered and inside the video')
        return start, end
    ground_truth = [span(t) for t in targets]
    def pair(pred, target):
        try:
            start, end = span(pred)
        except (ValueError, KeyError, TypeError):
            return 0.0
        if min(end, target[1]) <= max(start, target[0]):
            return 0.0
        return falloff(abs(start-target[0]), start_full, start_zero) * falloff(abs(end-target[1]), end_full, end_zero)
    weights = [[pair(p, t) for t in ground_truth] for p in predicted]
    return maximum_matching(weights) / max(len(predicted), len(targets))


def combined_reward(predicted, target, spatial=None, temporal=None, negative_arrays_empty=True, weights=None):
    if type(target) is not int or target not in (0, 1):
        raise ValueError('Target must be integer 0 or 1')
    if type(predicted) is not int or predicted != target:
        return 0.0
    if target == 0:
        return float(negative_arrays_empty)
    components = dict(answer=1.0)
    if spatial is not None:
        components['spatial'] = spatial
    if temporal is not None:
        components['temporal'] = temporal
    if len(components) == 1:
        return 1.0
    if weights is None:
        weights = dict(answer=.6, **{k: .4/(len(components)-1) for k in components if k != 'answer'})
    if weights.keys() != components.keys() or not math.isclose(sum(weights.values()), 1):
        raise ValueError('Weights must match components and sum to 1')
    if any(not number(v) or not 0 <= v <= 1 for v in (*weights.values(), *components.values())):
        raise ValueError('Reward components and weights must lie in [0, 1]')
    return sum(weights[k] * value for k, value in components.items())


def parse_prediction(text, schema):
    """Reject duplicate keys, nonfinite numbers, extra fields, and wrong JSON types."""
    def invalid_constant(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    Draft202012Validator(schema).validate(value)
    return value


def medical_reward(text, target, definition):
    """Validate category rules before scoring; invalid human targets raise errors."""
    check_answer(target, definition)
    try:
        predicted = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(predicted, dict):
            return 0.0
        check_answer(predicted, definition)
    except (ValueError, TypeError):
        return 0.0
    return selected_label_f1(predicted, target)


def exact_reward(text, target, schema):
    """Score viewpoint and single-label answers against their task schema."""
    from jsonschema.exceptions import ValidationError

    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    validator.validate(target)
    try:
        predicted = parse_prediction(text, schema)
    except (ValueError, TypeError, ValidationError):
        return 0.0
    return float(predicted == target)
