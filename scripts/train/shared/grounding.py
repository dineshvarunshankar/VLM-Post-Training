"""Convert per-frame annotations into counting and grounding training targets."""

import math


def timestamp(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and nonnegative")
    milliseconds = int(math.floor(seconds * 1000 + 0.5))
    minutes, remainder = divmod(milliseconds, 60_000)
    if minutes > 99:
        raise ValueError("Timestamp exceeds MM:SS format")
    base = f"{minutes:02d}:{remainder // 1000:02d}"
    fraction = f"{remainder % 1000:03d}".rstrip("0")
    return base + ("." + fraction if fraction else "")


def video_geometry(meta):
    fps, count = meta["fps"], meta["n_frames"]
    if (
        not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
        or type(count) is not int
        or count < 1
    ):
        raise ValueError("Invalid annotated video FPS or frame count")
    convention = meta.get("frame_convention", {})
    if (
        convention.get("first_frame") != 1
        or convention.get("time_s") != "(frame - 1) / fps"
    ):
        raise ValueError("Unsupported annotation frame convention")
    width, height = meta["resolution"]["width"], meta["resolution"]["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid annotation resolution")
    return fps, count, width, height


def frame_index(frame, meta):
    _, count, _, _ = video_geometry(meta)
    if type(frame) is not int or not 1 <= frame <= count:
        raise ValueError("Annotation frame is outside the video")
    return frame - 1


def valid_point(point, width, height):
    return (
        isinstance(point, list)
        and len(point) == 2
        and all(type(v) in (int, float) and math.isfinite(v) for v in point)
        and 0 <= point[0] <= width
        and 0 <= point[1] <= height
    )


def counting_targets(counting, points, meta):
    """Count only frames with human records for every relevant person's visibility."""
    fps, _, width, height = video_geometry(meta)
    people = counting.get("people", [])
    if len({p["id"] for p in people}) != len(people):
        raise ValueError("Counting contains duplicate person IDs")
    subject_type = counting.get("subject_type")
    expected = {
        f"person:{p['id']}"
        for p in people
        if p["role"] in ("other", "manikin")
        or (
            p["role"] == "subject"
            and subject_type in ("casualty_human_actor", "casualty_manikin")
        )
    }
    if not expected:
        return []
    result, seen = [], set()
    for row in points.get("frames", []):
        index = frame_index(row["frame"], meta)
        if index in seen:
            raise ValueError("Duplicate frame in counting tracks")
        seen.add(index)
        parts = row["parts"]
        states = {p["part"]: p for p in parts}
        if len(states) != len(parts):
            raise ValueError("Duplicate point identifier in frame")
        if not expected <= states.keys():
            continue
        count, complete = 0, True
        for identifier in expected:
            item = states[identifier]
            if item.get("by") != "human":
                complete = False
                break
            if item.get("occluded") is True:
                continue
            point = item.get("point")
            if item.get("occluded") is not False or point is None:
                complete = False
                break
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in point
                )
            ):
                complete = False
                break
            if valid_point(point, width, height):
                count += 1
        if complete:
            result.append(
                {"frame": index, "timestamp": timestamp(index / fps), "count": count}
            )
    return sorted(result, key=lambda row: row["frame"])


def human_evidence(cue, meta):
    """Select the required human seeds in annotation order, covering required slots."""
    fps, _, width, height = video_geometry(meta)
    if not meta['frame_convention'].get('coordinates', '').startswith('absolute pixels'):
        raise ValueError('Human evidence must declare absolute pixel coordinates')
    minimum = cue['evidence_min']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Spatial reward requires evidence_min > 0')
    slots = cue.get('evidence_slots') or []
    if slots and (len(slots) != minimum or len(set(slots)) != len(slots)):
        raise ValueError('Required slots must be unique and match evidence_min')
    # labels.json stores human seed points; points.json contains derived tracks.
    candidates = []
    for seed in cue['evidence_points']:
        frame = frame_index(seed['frame'], meta)
        xy = seed['point']
        if not valid_point(xy, width, height):
            raise ValueError('Human evidence is outside the annotated frame')
        result = dict(timestamp=timestamp(frame/fps), point=[1000*xy[0]/width, 1000*xy[1]/height], by='human', frame_index=frame)
        if seed.get('slot') is not None:
            result['slot'] = seed['slot']
        candidates.append(result)
    if slots:
        required = []
        for slot in slots:
            match = next((p for p in candidates if p.get('slot') == slot), None)
            if match is None:
                raise ValueError(f'Required slot lacks human seed evidence: {slot}')
            required.append(match)
    else:
        required = candidates[:minimum]
    if len(required) < minimum:
        raise ValueError('Required evidence lacks human seed points')
    return required



def evidence_requirements(cue):
    """Describe annotation requirements without consulting the selected answer."""
    minimum = cue['evidence_min']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Spatial question requires evidence_min > 0')
    slots = cue.get('evidence_slots') or []
    if (not isinstance(slots, list) or any(not isinstance(s, str) or not s for s in slots)
            or len(set(slots)) != len(slots) or (slots and len(slots) != minimum)):
        raise ValueError('Required slots must be unique and match evidence_min')
    text = f'If selected, provide {minimum} evidence point(s), each with its timestamp.'
    if slots:
        text += ' Include the slot name and one point for each required anatomical location: ' + ', '.join(slots) + '.'
    return text


def specialize_evidence_schema(schema, cue):
    """Required slots depend on label requirements, never on the human answer."""
    evidence_requirements(cue)
    slots = cue.get('evidence_slots') or []
    if slots:
        item = schema['properties']['evidence']['items']
        item['properties']['slot'] = dict(type='string', enum=slots)
        if 'slot' not in item['required']:
            item['required'].append('slot')


def spatial_target(cue, meta):
    """Build SFT/RL output from the same required human label seeds."""
    points = human_evidence(cue, meta)
    result = []
    for point in points:
        item = {k: v for k, v in point.items() if k in ('timestamp', 'point', 'slot')}
        item['point'] = [int(math.floor(v + .5)) for v in point['point']]
        result.append(item)
    return result, sorted({p['frame_index'] for p in points})


def require_matching_geometry(meta, geometry):
    """Accept millisecond-FPS serialization precision, not different media."""
    fps, count, width, height = video_geometry(meta)
    if (geometry is None or any(geometry.get(k) != v for k, v in
            dict(width=width, height=height, n_frames=count).items())
            or not math.isclose(geometry.get('fps', 0), fps, rel_tol=0, abs_tol=0.0005)):
        raise ValueError('Decoded geometry differs from annotation metadata')


def annotation_duration(meta):
    """Recover missing duration from the writer-validated frame geometry."""
    fps, count, _, _ = video_geometry(meta)
    duration = meta.get('duration_s')
    if duration is None:
        return count / fps
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid annotated video duration')
    return duration


def label_grounding_targets(cue, meta, *, spatial, temporal, on_issue=None):
    """Keep usable components; never turn missing positive evidence into absence."""
    target, frames = {}, []
    for enabled, key, task, build in (
        (spatial, 'evidence', 'spatial', spatial_target),
        (temporal, 'time_spans', 'temporal', temporal_target),
    ):
        if not enabled:
            continue
        try:
            values, selected = build(cue, meta) if cue['answer'] else ([], [])
            target[key] = values
            frames.extend(selected)
        except (ValueError, KeyError, TypeError) as error:
            if on_issue is not None:
                on_issue(task, error)
    return target, sorted(set(frames))


def temporal_duration_limit(meta):
    """Shared bound for target preparation and rewards, including serialization error."""
    fps, count, _, _ = video_geometry(meta)
    duration = annotation_duration(meta)
    # FPS is serialized to three decimals. Propagate that rounding uncertainty
    # only when duration has to be reconstructed; never alter annotated spans.
    if meta.get('duration_s') is None and fps > 0.0005:
        return count / (fps - 0.0005) + 0.001
    return duration + 0.001


def temporal_target(cue, meta):
    fps, count, _, _ = video_geometry(meta)
    end_limit = temporal_duration_limit(meta)
    spans = cue.get("time_spans", [])
    if not spans:
        raise ValueError("Positive temporal label has no annotated time spans")
    result, frames = [], set()
    for span in spans:
        start, end = span["start_s"], span["end_s"]
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (start, end)
            )
            or not (0 <= start < end and end <= end_limit)
        ):
            raise ValueError(
                "Annotated time span is outside the video or has invalid bounds"
            )
        formatted = {"start": timestamp(start), "end": timestamp(end)}
        if formatted["start"] == formatted["end"]:
            raise ValueError("Time span is shorter than output timestamp precision")
        if formatted not in result:
            result.append(formatted)
        for time in (start, (start + end) / 2, end):
            frames.add(min(count - 1, max(0, int(math.floor(time * fps + 0.5)))))
    return sorted(result, key=lambda x: (x["start"], x["end"])), sorted(frames)


def normalized_subject_box(box):
    """Preserve full box bounds while requiring a visible area in the image."""
    x, y, w, h = [box[k] for k in ('x', 'y', 'width', 'height')]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError('Invalid subject box coordinates')
    bounds = [10*x, 10*y, 10*(x+w), 10*(y+h)]
    if (w <= 0 or h <= 0 or not all(math.isfinite(v) for v in bounds)
            or max(0, bounds[0]) >= min(1000, bounds[2])
            or max(0, bounds[1]) >= min(1000, bounds[3])):
        raise ValueError('Subject box must have positive visible area inside the frame')
    return bounds


def subject_targets(counting, tracks, boxes, meta, *, on_issue=None):
    """Use visible human subject seeds and their same-frame subject boxes."""
    fps, _, width, height = video_geometry(meta)
    if not meta['frame_convention'].get('coordinates', '').startswith('absolute pixels'):
        raise ValueError('Subject points require absolute pixel coordinates')
    subjects = [p['id'] for p in counting['people'] if p['role'] == 'subject']
    if len(subjects) != 1:
        raise ValueError('Expected one designated subject in counting annotations')
    if not boxes.get('box_convention', '').startswith('bbox_pct is percent of frame, origin top-left'):
        raise ValueError('Subject boxes require the declared percent coordinate convention')
    by_frame = {}
    for row in boxes['frames']:
        if row['frame'] in by_frame:
            raise ValueError('Duplicate subject box frame')
        by_frame[row['frame']] = row.get('bbox_pct')
    result, seen = [], set()
    for row in tracks['frames']:
        frame = frame_index(row['frame'], meta)
        parts = [p for p in row['parts'] if p['part'] == f'person:{subjects[0]}']
        if len(parts) > 1 or frame in seen:
            raise ValueError('Duplicate subject point frame')
        seen.add(frame)
        if not parts or parts[0].get('by') != 'human' or parts[0].get('occluded') is not False:
            continue
        xy, box = parts[0]['point'], by_frame.get(row['frame'])
        if box is None:
            continue
        try:
            if not valid_point(xy, width, height):
                raise ValueError('Human subject point is outside the video frame')
            bounds = normalized_subject_box(box)
            normalized = [1000*xy[0]/width,1000*xy[1]/height]
            rounded = [int(math.floor(v+.5)) for v in normalized]
            if not all(bounds[axis] <= p[axis] <= bounds[axis+2] for p in (normalized,rounded) for axis in (0,1)):
                raise ValueError('Human subject point or its rounded target is outside the subject box')
        except (ValueError, KeyError, TypeError) as error:
            if on_issue is not None:
                on_issue(f"Frame {row['frame']}: {error}")
            continue
        result.append(dict(frame=frame,timestamp=timestamp(frame/fps),point=rounded,box=bounds))
    return sorted(result,key=lambda row:row['frame'])
