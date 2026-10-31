#!/usr/bin/env python3
"""Prepare RGB human-answer targets from currently completed annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.preprocess.completion_manifest import CompletionManifest, DEFAULT_MANIFEST

from scripts.train.shared.data import (
    DATASETS,
    atomic_json,
    extract_answers,
    read_json,
    read_yaml,
    sha256,
)


def clips(root):
    for variant, name in DATASETS.items():
        if (root / name).is_symlink():
            raise ValueError("Dataset folder must not be a symlink")
        for clip in sorted((root / name).glob("*")):
            if (
                clip.is_dir()
                and not clip.is_symlink()
                and (clip / "all_annotations").is_dir()
            ):
                yield variant, clip


def main():
    from dotenv import load_dotenv
    from scripts.train.shared.paths import REPO_ROOT
    from scripts.train.shared.storage_settings import local_dataset_directory, annotation_dataset_directory
    load_dotenv(REPO_ROOT / ".env")
    media_root = local_dataset_directory() or annotation_dataset_directory()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["gt"])
    parser.add_argument(
        "--media-root", type=Path, default=media_root, required=media_root is None, help="Parent of 06 and 07"
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--completed-manifest", type=Path, help="Current site completion CSV; defaults to media-root/00-data_website/medical/completed_clips.csv")
    parser.add_argument("--inspect", action="store_true", help="Report without writing")
    args = parser.parse_args()
    root = args.media_root.resolve()
    catalog_hash = sha256(args.catalog)
    catalog = read_yaml(args.catalog)
    if sha256(args.catalog) != catalog_hash:
        parser.error("Catalog changed during loading")
    try:
        manifest = CompletionManifest(args.completed_manifest or root / DEFAULT_MANIFEST, root)
        completion_snapshot = manifest.summary()
        eligible = {(variant, entry["clip_id"]): entry
                    for variant in DATASETS for entry in manifest.entries(variant)}
        if manifest.summary()["sha256"] != completion_snapshot["sha256"]:
            parser.error("Completion manifest changed during loading; retry preparation")
    except (OSError, ValueError) as error:
        parser.error(f"Cannot load current completion manifest: {error}")
    counts, errors = Counter(), []
    found = list(clips(root))
    if not found:
        parser.error(
            "No per-clip all_annotations folders found. Check --media-root and run preprocessing first."
        )
    for variant, clip in found:
        entry = eligible.get((variant, clip.name))
        if entry is None:
            counts["not_in_completion_manifest"] += 1
            continue
        try:
            meta = read_json(clip / "all_annotations/meta.json")
            if meta.get("clip") != entry["clip"]:
                raise ValueError("Prepared clip identity differs from completion manifest")
            if meta.get("modality") != "RGB":
                counts["ineligible"] += 1
                continue
            labels = clip / "all_annotations/labels.json"
            label_hash = sha256(labels)
            provenance = read_json(clip / "provenance.json")
            if (
                provenance.get("annotation_sha256", {}).get("labels.json")
                != label_hash
            ):
                raise ValueError("Copied labels differ from writer provenance")
            answers, excluded = extract_answers(read_json(labels), catalog, variant)
            for category, reason in excluded.items():
                errors.append(
                    {"clip": str(clip), "category": category, "reason": reason}
                )
            counts["excluded_categories"] += len(excluded)
            value = {
                "categories": answers,
                "source_hashes": {
                    "labels_sha256": label_hash,
                    "label_catalog_sha256": catalog_hash,
                },
            }
            if sha256(labels) != label_hash or sha256(args.catalog) != catalog_hash:
                raise ValueError("Labels changed during preparation")
            output = clip / "GT/medical_vqa.json"
            if output.parent.is_symlink() or not output.resolve().is_relative_to(
                clip.resolve()
            ):
                raise ValueError("GT destination escapes the clip folder")
            if output.exists() and read_json(output) == value:
                counts["unchanged"] += 1
            elif not args.inspect:
                atomic_json(output, value)
                counts["written"] += 1
            else:
                counts["would_write"] += 1
            counts["categories"] += len(answers)
        except (OSError, ValueError, KeyError, TypeError) as error:
            counts["failed"] += 1
            errors.append({"clip": str(clip), "reason": str(error)})
    if manifest.summary()["sha256"] != completion_snapshot["sha256"]:
        parser.error("Completion manifest changed during GT preparation; rerun the command")
    print(json.dumps({"counts": dict(counts), "issues": errors, "completion_manifest": completion_snapshot}, indent=2))
    return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
