"""Read human targets and construct independent, answer-only video examples."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

DATASETS = {
    "native": "06-native_media_dataset",
    "stabilized": "07-stabilized_media_dataset",
}
COMPLETE = {"completed", "completed_minimal"}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def label_keys(definition):
    return [
        cue["name"]
        for group in definition["subcategories"].values()
        for cue in group["cues"]
    ]


def check_answer(answer, definition):
    expected = label_keys(definition)
    if set(answer) != set(expected) or len(expected) != len(set(expected)):
        raise ValueError("Answer keys do not match the category catalog")
    if any(type(value) is not int or value not in (0, 1) for value in answer.values()):
        raise ValueError("Answers must be integer 0 or 1")
    observed = sum(
        answer[cue["name"]]
        for cue in definition["subcategories"].get("observables", {}).get("cues", [])
    )
    for group in definition["subcategories"].values():
        count = sum(answer[cue["name"]] for cue in group["cues"])
        rule = group["selection_rule"]
        valid = {
            "multi_select_or_none": True,
            "one_or_more": count >= 1,
            "select_one_or_none": count <= 1,
            "exactly_one": count == 1,
            "exactly_one_if_no_observables": count == (0 if observed else 1),
            "default_if_no_observables": count == (0 if observed else 1),
        }
        if rule not in valid or not valid[rule]:
            raise ValueError(f"Human answer does not satisfy selection rule: {rule}")


def category_enabled(category, variant):
    if variant not in DATASETS:
        raise ValueError(f"Unknown video variant: {variant}")
    return category != "subject_check" and (
        category != "context" or variant == "native"
    )


def extract_answers(labels, catalog, variant):
    answers, excluded = {}, {}
    for key, definition in catalog["categories"].items():
        if not category_enabled(key, variant):
            continue
        try:
            annotated = labels["categories"][definition["name"]]
            cue_map = {}
            for group in annotated.values():
                if isinstance(group, dict) and isinstance(group.get("cues"), dict):
                    if cue_map.keys() & group["cues"].keys():
                        raise ValueError("Duplicate human cue keys within category")
                    cue_map.update(group["cues"])
            answer = {}
            for group_name, group in definition["subcategories"].items():
                for cue in group["cues"]:
                    name = cue["name"]
                    item = cue_map[name]
                    if (
                        item.get("answered") is not True
                        or type(item.get("answer")) is not bool
                    ):
                        raise ValueError(f"Missing answered human label: {name}")
                    answer[name] = int(item["answer"])
            check_answer(answer, definition)
            answers[key] = answer
        except (KeyError, TypeError, ValueError) as error:
            excluded[key] = str(error)
    return answers, excluded


def recording_group(meta):
    # The source recording precedes the person/segment/clip suffix in annotation IDs.
    match = re.fullmatch(r"(.+)_i\d+s\d+_[CF]\d+(?:_S)?", meta["clip"])
    if not match:
        raise ValueError(f"Cannot establish recording group from {meta['clip']!r}")
    return match.group(1)


def clip_path(root, variant, clip_id):
    if (
        variant not in DATASETS
        or not isinstance(clip_id, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id)
    ):
        raise ValueError("Invalid variant or clip folder ID")
    dataset = Path(root).resolve() / DATASETS[variant]
    path = dataset / clip_id
    if path.is_symlink() or not path.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Clip path escapes its dataset")
    return path


def reject_pairs(root):
    rejected = set()
    for name in DATASETS.values():
        dataset = Path(root) / name
        for path in dataset.glob("*/all_annotations/rejected.json"):
            meta = read_json(path.parent / "meta.json")
            rejected.add(meta["clip"].removesuffix("_S"))
    # Include current rejection markers when original sources share the mounted root.
    for name in ("04-segmented_snippets", "05-segmented_stabilized_snippets"):
        for path in (Path(root) / name / "00-medical_annotations").glob(
            "*/rejected.json"
        ):
            rejected.add(path.parent.name.removesuffix("_S"))
    return rejected


def load_clip(clip, catalog_path, rejected, completion_approved=False):
    annotations = clip / "all_annotations"
    meta = read_json(annotations / "meta.json")
    if not completion_approved and ((annotations / "rejected.json").exists() or meta["clip"].removesuffix(
        "_S"
    ) in rejected):
        raise ValueError("Rejected clip or counterpart")
    if meta.get("modality") != "RGB" or (not completion_approved and meta.get("annotation_status") not in COMPLETE):
        raise ValueError("Requires RGB and completed_minimal/completed status")
    video = clip / (clip.name + ".mp4")
    if not video.is_file() or Path(meta["video_path"]).name != video.name:
        raise ValueError(
            "Prepared video is missing or does not match annotation identity"
        )
    gt = read_json(clip / "GT/medical_vqa.json")
    if gt.get("source_hashes") != {
        "labels_sha256": sha256(annotations / "labels.json"),
        "label_catalog_sha256": sha256(catalog_path),
    }:
        raise ValueError("GT is stale; regenerate it from current labels and catalog")
    provenance = read_json(clip / "provenance.json")
    if (
        provenance.get("annotation_sha256", {}).get("labels.json")
        != gt["source_hashes"]["labels_sha256"]
    ):
        raise ValueError(
            "Copied labels differ from writer provenance; rerun preprocessing"
        )
    for name in ("meta.json", "camera_motion.json"):
        if sha256(annotations / name) != provenance.get("annotation_sha256", {}).get(
            name
        ):
            raise ValueError(
                f"{name} differs from writer provenance; rerun preprocessing"
            )
    return meta, gt, recording_group(meta)


def category_prompt_values(definition, templates, viewpoint):
    definitions = "\n\n".join(
        templates["subcategory_template"].format(
            subcategory_name=name.upper(),
            cues="\n".join(
                templates["label_template"].format(
                    label=cue["name"], definition=cue["definition"]
                )
                for cue in group["cues"]
            ),
        )
        for name, group in definition["subcategories"].items()
    )
    rules = "\n".join(
        templates["selection_rule_template"].format(
            subcategory_name=name.upper(),
            selection_rule_instruction=templates["selection_rule_instructions"][
                group["selection_rule"]
            ],
        )
        for name, group in definition["subcategories"].items()
    )
    assessment_names = [
        cue["name"]
        for cue in definition["subcategories"].get("assessment", {}).get("cues", [])
    ]
    absent = [
        name
        for name in assessment_names
        if name.endswith(("_assessed_absent", "_assessed_no_injury"))
    ]
    not_assessable = [
        name for name in assessment_names if name.endswith("_not_assessable")
    ]
    if absent and (len(absent) != 1 or len(not_assessable) != 1):
        raise ValueError(
            "Assessment instruction requires one absent and one not-assessable label"
        )
    assessment = (
        "\n"
        + templates["assessment_instruction"].format(
            absent_label=absent[0], not_assessable_label=not_assessable[0]
        )
        if absent
        else ""
    )
    if not isinstance(viewpoint, str) or not viewpoint.strip():
        raise ValueError("Camera viewpoint is missing")
    angle = {
        "aerial_top_down": "nadir",
        "aerial_oblique": "oblique",
        "nadir": "nadir",
        "oblique": "oblique",
    }.get(viewpoint)
    note = (
        templates["camera_angle_templates"][angle]
        if angle
        else "CAMERA VIEWPOINT\n" + viewpoint.replace("_", " ")
    )
    return {
        "category_name": definition["name"],
        "definitions": definitions,
        "selection_rules": rules,
        "assessment_instruction": assessment,
        "viewpoint": note.removeprefix("CAMERA VIEWPOINT\n").removeprefix(
            "CAMERA ANGLE\n"
        ),
        "viewpoint_note": note,
    }


def render_prompt(category, definition, templates, viewpoint):
    values = category_prompt_values(definition, templates, viewpoint)
    text = templates["category_template"].format(
        **values,
        task_instruction=templates.get("task_templates", {}).get("single_turn", ""),
        label_keys=json.dumps(label_keys(definition)),
    )
    system = templates["system_templates"][
        "subject_check" if category == "subject_check" else "assessment"
    ]
    return system, values["viewpoint_note"] + "\n\n" + text


def frozen_split_membership(split_dir, catalog_path):
    """Use the approved completion snapshot without consulting mutable source exports."""
    path = Path(split_dir) / 'summary.json'
    if not path.exists():
        return None
    summary = read_json(path)
    if (summary.get('schema_version') != 1 or
            summary.get('policy', {}).get('grouping') != 'original_source_recording'):
        raise ValueError('Unsupported split summary format')
    if summary.get('modality') != 'RGB':
        raise ValueError('Current training recipes support RGB splits only')
    if summary.get('catalog_sha256') != sha256(catalog_path):
        raise ValueError('Split catalog changed; review and regenerate the split')
    membership = {}
    for split in ('train', 'validation', 'test'):
        if sha256(Path(split_dir) / (split + '.jsonl')) != summary['split_sha256'][split]:
            raise ValueError(f'{split}.jsonl differs from its frozen split summary')
        membership[split] = {}
        for pair in summary['pairs'][split]:
            for variant in DATASETS:
                key = (variant, pair[variant + '_id'])
                if key in membership[split]:
                    raise ValueError('Duplicate clip in frozen split membership')
                membership[split][key] = pair['recording']
    return membership


def load_splits(root, split_dir, catalog_path, prompts_path, sources=None, evaluation_sources=None):
    catalog, templates = read_yaml(catalog_path), read_yaml(prompts_path)
    membership = frozen_split_membership(split_dir, catalog_path)
    rejected = set() if membership is not None else reject_pairs(root)
    cache, result, owners, seen = {}, {}, {}, set()
    for split in ("train", "validation", "test"):
        split_sources = sources
        if split != "train" and evaluation_sources is not None:
            split_sources = {name: name in evaluation_sources for name in DATASETS}
        rows = []
        path = Path(split_dir) / (split + ".jsonl")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            ref = json.loads(line)
            variant, clip_id, category = ref["variant"], ref["clip"], ref["category"]
            if split_sources is not None and variant in DATASETS and not split_sources[variant]:
                continue
            if not category_enabled(category, variant):
                raise ValueError(
                    f"Category {category} is excluded from {variant} SFT; regenerate the references"
                )
            key = (variant, clip_id, category)
            if key in seen:
                raise ValueError(f"Duplicate example across split files: {key}")
            seen.add(key)
            approved = membership is not None
            if approved and (variant, clip_id) not in membership[split]:
                raise ValueError('Example is not in the frozen split membership')
            clip = clip_path(root, variant, clip_id)
            if clip not in cache:
                meta, gt, group = load_clip(clip, catalog_path, rejected, completion_approved=approved)
                camera = read_json(clip / "all_annotations/camera_motion.json")
                cache[clip] = (
                    meta,
                    gt,
                    group,
                    camera["view_type"],
                    read_json(clip / "provenance.json").get("output_sha256"),
                )
            meta, gt, group, viewpoint, video_hash = cache[clip]
            if approved:
                group = membership[split][(variant, clip_id)]
            if group in owners and owners[group] != split:
                raise ValueError(
                    "Source recording appears in multiple splits: " + group
                )
            owners[group] = split
            answer = gt["categories"][category]
            definition = catalog["categories"][category]
            check_answer(answer, definition)
            system, question = render_prompt(category, definition, templates, viewpoint)
            rows.append(
                {
                    "video": str(clip / (clip_id + ".mp4")),
                    "system": system,
                    "question": question,
                    "answer": json.dumps(answer, separators=(",", ":")),
                    "reference": ref,
                    "video_sha256": video_hash,
                    "video_metadata": {"fps": meta["fps"], "n_frames": meta["n_frames"]},
                }
            )
        result[split] = rows
    if not result["train"] or not result["validation"]:
        raise ValueError("Training and validation splits must be nonempty")
    return result
