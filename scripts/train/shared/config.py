"""Load inherited experiment settings and validate task sampling configuration."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml
from scripts.train.shared.data import read_yaml


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_training_config(path, _parents=()):
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    config = read_yaml(path)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    parent = config.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str) or not parent:
            raise ValueError("extends must name a parent YAML file")
        config = merge_config(
            load_training_config(path.parent / parent, (*_parents, path)), config
        )
    validate_augmentation(config)
    validate_mixture(config)
    return config


def fields(value, required, location):
    if not isinstance(value, dict) or set(value) != set(required.split()):
        raise ValueError(f"{location} must contain exactly: {required}")


def boolean(value, location):
    if type(value) is not bool:
        raise ValueError(f"{location} must be true or false")


def choices(value, allowed, location):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{location} must be one of: {', '.join(allowed)}")


def string_list(value, location, allowed=None):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError(f"{location} must be a list of strings")
    if len(value) != len(set(value)) or (
        allowed and any(x not in allowed for x in value)
    ):
        raise ValueError(f"{location} contains duplicate or unsupported values")


def validate_augmentation(config):
    """Keep staged augmentation settings explicit and inactive until implemented."""
    augmentation = config.get("augmentation")
    if augmentation is None:
        return
    fields(augmentation, "enabled photometric horizontal_flip", "augmentation")
    boolean(augmentation["enabled"], "augmentation.enabled")
    flip = augmentation["horizontal_flip"]
    fields(flip, "enabled", "augmentation.horizontal_flip")
    boolean(flip["enabled"], "augmentation.horizontal_flip.enabled")
    photo = augmentation["photometric"]
    fields(photo, "probability brightness contrast", "augmentation.photometric")
    probability = photo["probability"]
    if type(probability) not in (int, float) or not 0 <= probability <= 1:
        raise ValueError("augmentation.photometric.probability must be between 0 and 1")
    for name in ("brightness", "contrast"):
        bounds = photo[name]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(value) not in (int, float) for value in bounds)
                or not 0 < bounds[0] <= bounds[1] < float("inf")):
            raise ValueError(f"augmentation.photometric.{name} requires ordered positive finite multipliers")
    if augmentation["enabled"] or flip["enabled"]:
        raise ValueError(
            "Video augmentation is configured but not implemented; keep enabled flags false "
            "until clip-consistent transforms, overlay preservation, and GT alignment are verified"
        )


def validate_mixture(config):
    if "dataset" not in config and "mixture" not in config:
        return
    dataset, mixture = config.get("dataset"), config.get("mixture")
    fields(dataset, "modality sources", "dataset")
    choices(dataset["modality"], ["RGB"], "dataset.modality")
    fields(dataset["sources"], "native stabilized", "dataset.sources")
    for source, enabled in dataset["sources"].items():
        boolean(enabled, f"dataset.sources.{source}")
    if not any(dataset["sources"].values()):
        raise ValueError("Enable at least one dataset source")
    fields(
        mixture,
        "strategy shuffle resample_each_epoch medical_vqa camera_view counting subject_localization label_questions",
        "mixture",
    )
    choices(mixture["strategy"], ["all_medical_plus_auxiliary"], "mixture.strategy")
    for key in ("shuffle", "resample_each_epoch"):
        boolean(mixture[key], f"mixture.{key}")
    medical = mixture["medical_vqa"]
    fields(
        medical, "enabled sampling excluded_categories context_sources", "medical_vqa"
    )
    choices(medical["sampling"], ["all_valid"], "medical_vqa.sampling")
    string_list(medical["excluded_categories"], "medical_vqa.excluded_categories")
    string_list(medical["context_sources"], "medical_vqa.context_sources", ["native"])
    for name in ("camera_view", "counting", "subject_localization"):
        task = mixture[name]
        required = "enabled sources questions_per_video_per_epoch"
        if name == "counting":
            required += " timestamp_sampling"
        fields(task, required, name)
        string_list(task["sources"], f"{name}.sources", ["native", "stabilized"])
        count = task["questions_per_video_per_epoch"]
        if type(count) is not int or count < 1:
            raise ValueError(
                f"{name}.questions_per_video_per_epoch must be a positive integer"
            )
        if task["enabled"] and not any(dataset["sources"][s] for s in task["sources"]):
            raise ValueError(f"{name} has no enabled dataset source")
    choices(
        mixture["counting"]["timestamp_sampling"],
        ["random_annotated"],
        "counting.timestamp_sampling",
    )
    labels = mixture["label_questions"]
    fields(labels, "enabled excluded_categories sampling grounding", "label_questions")
    string_list(labels["excluded_categories"], "label_questions.excluded_categories")
    sampling = labels["sampling"]
    fields(
        sampling,
        "observables assessment single_subcategory prefer_positive_negative_pair without_replacement cap_to_available_labels",
        "label_questions.sampling",
    )
    for name in ("observables", "assessment", "single_subcategory"):
        if type(sampling[name]) is not int or sampling[name] < 1:
            raise ValueError(
                f"label_questions.sampling.{name} must be a positive integer"
            )
    for name in (
        "prefer_positive_negative_pair",
        "without_replacement",
        "cap_to_available_labels",
    ):
        boolean(sampling[name], f"label_questions.sampling.{name}")
    fields(labels["grounding"], "spatial temporal", "label_questions.grounding")
    fields(labels["grounding"]["spatial"], "enabled", "grounding.spatial")
    fields(
        labels["grounding"]["temporal"],
        "enabled temporal_labels_only",
        "grounding.temporal",
    )
    for name in ("spatial", "temporal"):
        boolean(labels["grounding"][name]["enabled"], f"grounding.{name}.enabled")
    boolean(
        labels["grounding"]["temporal"]["temporal_labels_only"], "temporal_labels_only"
    )
    if (
        labels["grounding"]["temporal"]["enabled"]
        and not labels["grounding"]["temporal"]["temporal_labels_only"]
    ):
        raise ValueError(
            "Temporal grounding is supported only for labels annotated temporal=true"
        )
    for name in ("medical_vqa", "camera_view", "counting", "subject_localization", "label_questions"):
        boolean(mixture[name]["enabled"], f"{name}.enabled")
    if not any(
        mixture[name]["enabled"]
        for name in ("medical_vqa", "camera_view", "counting", "subject_localization", "label_questions")
    ):
        raise ValueError("Enable at least one training task")


def select_medical_examples(splits, config):
    if "mixture" not in config:
        return splits
    sources = config["dataset"]["sources"]
    policy = config["mixture"]["medical_vqa"]
    selected = {}
    for split, rows in splits.items():
        selected[split] = [
            row
            for row in rows
            if sources[row["reference"]["variant"]]
            and row["reference"]["category"] not in policy["excluded_categories"]
            and (
                row["reference"]["category"] != "context"
                or row["reference"]["variant"] in policy["context_sources"]
            )
        ]
    if not selected["train"] or not selected["validation"]:
        raise ValueError(
            "Configured sources and categories must leave nonempty training and validation splits"
        )
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(yaml.safe_dump(load_training_config(args.config), sort_keys=False))


def evaluation_settings(config, model_profile):
    """Resolve the same validation decoding and answer budget for SFT and RL."""
    limit = config['evaluation']['max_new_tokens']
    if type(limit) is not int or limit <= 0:
        raise ValueError('evaluation.max_new_tokens must be a positive integer')
    return {'generation': deepcopy(model_profile['generation']), 'max_new_tokens': limit}


def validate_grounding_readiness(config):
    """Require explicit dataset verification before using stabilized spatial targets."""
    verified = config.get("data_checks", {}).get("stabilized_coordinates_verified", False)
    boolean(verified, "data_checks.stabilized_coordinates_verified")
    if not config.get("dataset", {}).get("sources", {}).get("stabilized", False):
        return
    mixture = config.get("mixture", {})
    subject = mixture.get("subject_localization", {})
    labels = mixture.get("label_questions", {})
    subject_enabled = subject.get("enabled", False) and "stabilized" in subject.get("sources", [])
    evidence_enabled = labels.get("enabled", False) and labels.get("grounding", {}).get("spatial", {}).get("enabled", False)
    if (subject_enabled or evidence_enabled) and not verified:
        raise ValueError(
            "Stabilized spatial tasks require data_checks.stabilized_coordinates_verified: true. "
            "Verify subject and evidence coordinates against the corrected stabilized frames first; "
            "otherwise disable the affected tasks or stabilized source."
        )
