"""Build immutable task candidates from each video's copied human annotations."""


import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.train.shared.paths import ANSWER_ONLY_PROMPT_DIR
from scripts.train.shared.data import category_prompt_values, read_json, read_yaml, sha256
from scripts.train.shared.grounding import counting_targets, spatial_target, temporal_target, subject_targets, evidence_requirements
from scripts.train.shared.mixture import EpochMixtureSampler
from scripts.train.shared.spatial_policy import spatial_rule

PROMPT_DIRECTORY = ANSWER_ONLY_PROMPT_DIR / "RGB"
PROMPTS = (
    "camera_view",
    "subject_localization",
    "counting",
    "label_selection",
    "label_selection_grounding",
    "label_selection_temporal",
    "label_selection_spatiotemporal",
)


def build_mixture(
    medical_rows, catalog, medical_templates, config, prompt_directory=PROMPT_DIRECTORY, *, spatial_policy=None
):
    from scripts.train.shared.config import validate_grounding_readiness
    validate_grounding_readiness(config)
    policy = config["mixture"]
    prompts = {name: read_yaml(prompt_directory / (name + ".yaml")) for name in PROMPTS}
    pool, groups, issues = [], [], []
    plan = {"fixed": [], "groups": groups}
    videos = defaultdict(list)
    indices_by_video = defaultdict(list)
    for row in medical_rows:
        videos[row["video"]].append(row)
        if policy["medical_vqa"]["enabled"]:
            fixed = dict(row, required_frame_indices=[], task="medical_vqa")
            fixed["reference"] = dict(
                row["reference"], task="medical_vqa", label=None, timestamp=None
            )
            indices_by_video[row["video"]].append(len(pool))
            plan["fixed"].append(len(pool))
            pool.append(fixed)

    def issue(base, task, error, label=None):
        issues.append(
            {
                "variant": base["reference"]["variant"],
                "clip": base["reference"]["clip"],
                "task": task,
                "label": label,
                "reason": str(error),
            }
        )

    def add(base, task, answer, values, frames=(), label=None):
        template = prompts[task]
        meta = current_meta
        budget = (
            meta["n_frames"]
            if meta["fps"] <= config["video"]["fps"]
            else max(1, int(meta["n_frames"] / meta["fps"] * config["video"]["fps"]))
        )
        if len(set(frames)) > budget:
            raise ValueError(
                "Required grounding frames exceed the configured video frame budget"
            )
        row = dict(
            base,
            system=template["system_prompt"],
            question=template["user_prompt"].format(**values),
            answer=json.dumps(answer, separators=(",", ":")),
            required_frame_indices=sorted(set(frames)),
            task=task,
        )
        row["reference"] = dict(
            base["reference"],
            task=task,
            label=label,
            timestamp=values.get("timestamp"),
            category=base["reference"]["category"]
            if task.startswith("label_selection")
            else None,
        )
        indices_by_video[row["video"]].append(len(pool))
        pool.append(row)
        return len(pool) - 1

    def group(indices, values, count, balanced=False, replace=False, cap=True):
        if not indices:
            return
        if cap:
            count = min(count, len(indices))
        groups.append(
            {
                "indices": indices,
                "values": values,
                "count": count,
                "balanced": balanced,
                "replace": replace,
            }
        )

    for video, rows in sorted(videos.items()):
        base, variant = rows[0], rows[0]["reference"]["variant"]
        folder = Path(video).parent
        provenance = read_json(folder / "provenance.json")
        annotation_hashes = {}

        def annotation(name):
            path = folder / "all_annotations" / name
            actual = sha256(path)
            if provenance.get("annotation_sha256", {}).get(name) != actual:
                raise ValueError(
                    f"{name} differs from writer provenance; refresh preprocessing"
                )
            annotation_hashes[name] = actual
            return read_json(path)

        meta = annotation("meta.json")
        current_meta = meta
        viewpoint = annotation("camera_motion.json")["view_type"]
        camera = policy["camera_view"]
        if camera["enabled"] and variant in camera["sources"]:
            canonical = {"nadir": "aerial_top_down", "oblique": "aerial_oblique"}.get(
                viewpoint, viewpoint
            )
            choices = prompts["camera_view"]["response_schema"]["required"]
            if canonical not in choices:
                issue(base, "camera_view", "Unknown annotated viewpoint")
            else:
                index = add(
                    base,
                    "camera_view",
                    {name: int(name == canonical) for name in choices},
                    {},
                )
                group([index], [1], camera["questions_per_video_per_epoch"])
        counting = policy["counting"]
        if counting["enabled"] and variant in counting["sources"]:
            try:
                candidates = counting_targets(
                    annotation("counting.json"), annotation("points.json"), meta
                )
                indices = [
                    add(
                        base,
                        "counting",
                        {"count": c["count"]},
                        {"timestamp": c["timestamp"]},
                        [c["frame"]],
                    )
                    for c in candidates
                ]
                group(
                    indices,
                    [None] * len(indices),
                    counting["questions_per_video_per_epoch"],
                )
                if not indices:
                    issue(
                        base,
                        "counting",
                        "No frames with complete person visibility annotations",
                    )
            except (ValueError, KeyError, TypeError, OSError) as error:
                issue(base, "counting", error)
        subject = policy["subject_localization"]
        if subject["enabled"] and variant in subject["sources"]:
            try:
                candidates = subject_targets(
                    annotation("counting.json"), annotation("points.json"),
                    annotation("boxes.json"), meta,
                )
                indices = [
                    add(base, "subject_localization", {"point": c["point"]},
                        {"timestamp": c["timestamp"]}, [c["frame"]])
                    for c in candidates
                ]
                group(indices, [None] * len(indices), subject["questions_per_video_per_epoch"])
                if not indices:
                    issue(base, "subject_localization", "No visible human subject seed with a same-frame box")
            except (ValueError, KeyError, TypeError, OSError) as error:
                issue(base, "subject_localization", error)
        labels = policy["label_questions"]
        if labels["enabled"]:
            annotations = annotation("labels.json")
            cue_map = {}
            for category in annotations["categories"].values():
                for subcategory in category.values():
                    if isinstance(subcategory, dict):
                        for name, cue in subcategory.get("cues", {}).items():
                            if name in cue_map:
                                raise ValueError(
                                    "Duplicate cue in label annotations: " + name
                                )
                            cue_map[name] = cue
            for medical in sorted(rows, key=lambda row: row["reference"]["category"]):
                category = medical["reference"]["category"]
                if category in labels["excluded_categories"]:
                    continue
                definition = catalog["categories"][category]
                answer = json.loads(medical["answer"])
                values = category_prompt_values(
                    definition, medical_templates, viewpoint
                )
                for subcategory, details in definition["subcategories"].items():
                    indices, binary = [], []
                    for entry in details["cues"]:
                        name = entry["name"]
                        try:
                            cue = cue_map[name]
                            if (
                                cue.get("answered") is not True
                                or type(cue.get("answer")) is not bool
                                or int(cue["answer"]) != answer[name]
                            ):
                                raise ValueError(
                                    "Human label differs from prepared medical answer"
                                )
                            if (
                                type(cue.get("temporal")) is not bool
                                or type(cue.get("evidence_min")) is not int
                            ):
                                raise ValueError(
                                    "Cue lacks explicit temporal/evidence requirements"
                                )
                            rule = spatial_rule(spatial_policy, medical["reference"]["category"], name)
                            spatial = (
                                labels["grounding"]["spatial"]["enabled"]
                                and cue["evidence_min"] > 0
                                and rule["mode"] != "disabled"
                            )
                            temporal = (
                                labels["grounding"]["temporal"]["enabled"]
                                and cue["temporal"]
                            )
                            task = (
                                "label_selection_spatiotemporal"
                                if spatial and temporal
                                else "label_selection_grounding"
                                if spatial
                                else "label_selection_temporal"
                                if temporal
                                else "label_selection"
                            )
                            target = (
                                {name: answer[name]}
                                if not (spatial or temporal)
                                else {"answer": {name: answer[name]}}
                            )
                            frames = []
                            if spatial:
                                evidence, selected_frames = (
                                    spatial_target(cue, meta)
                                    if answer[name]
                                    else ([], [])
                                )
                                target["evidence"] = evidence
                                frames.extend(selected_frames)
                            if temporal:
                                spans, selected_frames = (
                                    temporal_target(cue, meta)
                                    if answer[name]
                                    else ([], [])
                                )
                                target["time_spans"] = spans
                                frames.extend(selected_frames)
                            requirements = ""
                            if spatial:
                                placement = ("Place the point on the designated subject."
                                             if rule["mode"] == "subject_box"
                                             else rule.get("instruction", ""))
                                requirements = (evidence_requirements(cue) + " " + placement).strip()
                            indices.append(
                                add(
                                    medical,
                                    task,
                                    target,
                                    dict(values, label=name, evidence_requirements=requirements),
                                    frames,
                                    name,
                                )
                            )
                            binary.append(answer[name])
                        except (KeyError, TypeError, ValueError) as error:
                            issue(medical, "label_questions", error, name)
                    sampling = labels["sampling"]
                    count = (
                        sampling["single_subcategory"]
                        if len(definition["subcategories"]) == 1
                        else sampling[subcategory]
                    )
                    group(
                        indices,
                        binary,
                        count,
                        sampling["prefer_positive_negative_pair"],
                        not sampling["without_replacement"],
                        sampling["cap_to_available_labels"],
                    )
        # Fingerprint every annotation used, including tracks; resume detects changes.
        for index in indices_by_video[video]:
            pool[index]["annotation_hashes"] = annotation_hashes
            pool[index]["video_metadata"] = {
                "fps": meta["fps"],
                "n_frames": meta["n_frames"],
            }
    sampler = EpochMixtureSampler(
        plan,
        seed=config["training"].get("data_seed")
        if config["training"].get("data_seed") is not None
        else config["training"]["seed"],
        shuffle=policy["shuffle"],
        resample=policy["resample_each_epoch"],
    )
    by_task, by_source = Counter(), Counter()
    for index in sampler:
        by_task[pool[index]["task"]] += 1
        by_source[pool[index]["reference"]["variant"]] += 1
    report = {
        "candidate_examples": len(pool),
        "examples_per_epoch": len(sampler),
        "epoch_zero_tasks": dict(by_task),
        "epoch_zero_sources": dict(by_source),
        "excluded_candidates": issues,
        "task_prompt_hashes": {
            name: sha256(prompt_directory / (name + ".yaml")) for name in PROMPTS
        },
    }
    return pool, plan, report
