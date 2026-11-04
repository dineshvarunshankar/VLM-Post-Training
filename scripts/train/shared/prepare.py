#!/usr/bin/env python3
"""Prepare RGB human-answer targets from currently completed annotations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.preprocess.prepared_manifest import PreparedManifest

from scripts.train.shared.data import (
    DATASETS,
    atomic_json,
    extract_answers,
    read_json,
    read_yaml,
    sha256,
)


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
    parser.add_argument("--include-subject-check", action="store_true", help="Also prepare human-reviewed subject-check image targets")
    parser.add_argument("--inspect", action="store_true", help="Report without writing")
    args = parser.parse_args()
    root = args.media_root.resolve()
    catalog_hash = sha256(args.catalog)
    catalog = read_yaml(args.catalog)
    if sha256(args.catalog) != catalog_hash:
        parser.error("Catalog changed during loading")
    try:
        manifest = PreparedManifest(root)
        prepared_snapshot = manifest.summary()
        eligible = {(variant, entry["clip_id"]): entry
                    for variant in DATASETS for entry in manifest.entries(variant)}
        manifest.verify_unchanged()
    except (OSError, ValueError) as error:
        parser.error(f"Cannot load prepared dataset manifests: {error}")
    counts, errors = Counter(), []
    found = [(variant, root / DATASETS[variant] / clip_id) for variant, clip_id in sorted(eligible)]
    if not found:
        parser.error(
            "No per-clip all_annotations folders found. Check --media-root and run preprocessing first."
        )
    for variant, clip in found:
        entry = eligible[(variant, clip.name)]
        try:
            if clip.is_symlink() or not clip.resolve().is_relative_to(root):
                raise ValueError("Prepared clip folder must remain inside the dataset")
            meta = read_json(clip / "all_annotations/meta.json")
            if meta.get("clip") != entry["clip"]:
                raise ValueError("Prepared clip identity differs from dataset manifest")
            if meta.get("modality") != "RGB":
                counts["ineligible"] += 1
                continue
            labels = clip / "all_annotations/labels.json"
            label_hash = sha256(labels)
            provenance_path = clip / "provenance.json"
            if sha256(provenance_path) != entry['provenance_sha256']:
                raise ValueError('Output changed since prepared manifest publication; finish preprocessing first')
            provenance = read_json(provenance_path)
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
            if args.include_subject_check and "subject_type" in answers:
                from scripts.train.shared.subject_check import prepare_video_subject_gt
                status = prepare_video_subject_gt(clip, answers["subject_type"], value["source_hashes"], inspect=args.inspect)
                counts["subject_check_" + status] += 1
            counts["categories"] += len(answers)
        except (OSError, ValueError, KeyError, TypeError) as error:
            counts["failed"] += 1
            errors.append({"clip": str(clip), "reason": str(error)})
    try:
        manifest.verify_unchanged()
    except (OSError, ValueError) as error:
        counts['failed'] += 1
        errors.append({'reason': str(error)})
    subject_report = None
    if args.include_subject_check:
        from scripts.train.shared.subject_check import prepare_negative_gt
        try:
            subject_report = prepare_negative_gt(root, inspect=args.inspect)
        except (OSError, ValueError, KeyError, TypeError) as error:
            counts["failed"] += 1
            errors.append({"reason": str(error)})
    print(json.dumps({"subject_check": subject_report, "counts": dict(counts), "issues": errors, "prepared_manifests": prepared_snapshot}, indent=2))
    return int(bool(counts["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
