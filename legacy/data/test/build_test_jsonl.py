#!/usr/bin/env python3
"""
Build data/test/<your_test_split>/exports/test.jsonl from category exports + selection_manifest.json.

Edit train_split / test_split below, then run from repo root:
  PYTHONPATH=. python data/test/build_test_jsonl.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --- edit these ---
root = Path(__file__).resolve().parents[2]
train_split = "your_train_split"
test_split = "your_test_split"
cot_jsonl = root / "data" / "train" / train_split / "cot_annotations" / "sft.jsonl"

EXPORTS_DIR = root / "data" / "test" / test_split / "exports"
OUT_JSONL = EXPORTS_DIR / "test.jsonl"
MANIFEST_FILENAME = "selection_manifest.json"

IMAGE_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
    ".JPG", ".JPEG", ".PNG",
)


def load_label_to_question(cot_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not cot_path.is_file():
        print(f"Warning: COT file not found: {cot_path}", file=sys.stderr)
        return out
    with cot_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            q = row.get("question")
            if label and q and label not in out:
                out[label] = q
    return out


def category_label_from_dirname(dirname: str) -> str | None:
    suffix = "-data-vf"
    if dirname.endswith(suffix):
        return dirname[: -len(suffix)]
    return None


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix in IMAGE_SUFFIXES


def parse_manifest(manifest_path: Path, category_dir: Path) -> dict[str, list[Path]] | None:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: bad manifest {manifest_path}: {e}", file=sys.stderr)
        return None

    out: dict[str, list[Path]] = {"yes": [], "no": []}

    def resolve_one(class_name: str, entry: str) -> Path | None:
        entry = entry.replace("\\", "/").strip()
        if not entry:
            return None
        parts = Path(entry).parts
        if len(parts) >= 2 and parts[0].lower() in ("yes", "no"):
            base = parts[0].lower()
            rel = Path(*parts[1:])
            cand = category_dir / base / rel
            return cand if cand.is_file() else None
        if class_name in ("yes", "no"):
            cand = category_dir / class_name / entry
            if cand.is_file():
                return cand
            d = category_dir / class_name
            if d.is_dir():
                matches = [x for x in d.iterdir() if x.name == entry and _is_image(x)]
                if len(matches) == 1:
                    return matches[0]
        return None

    if isinstance(raw, dict):
        if "yes" in raw or "no" in raw:
            for cls in ("yes", "no"):
                items = raw.get(cls)
                if items is None or not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict):
                        path_s = item.get("path") or item.get("image") or item.get("file")
                        if not path_s:
                            continue
                        ic = str(item.get("class", item.get("label_class", cls))).lower()
                        if ic not in ("yes", "no"):
                            ic = cls
                        r = resolve_one(ic, str(path_s))
                        if r and _is_image(r):
                            out[ic].append(r)
                    else:
                        r = resolve_one(cls, str(item))
                        if r and _is_image(r):
                            out[cls].append(r)
            return out if (out["yes"] or out["no"]) else None

        flat = raw.get("files") or raw.get("images") or raw.get("selection")
        if isinstance(flat, list):
            raw = flat
        else:
            return None

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                path_s = item.get("path") or item.get("image") or item.get("file")
                cls = str(item.get("class", item.get("split", ""))).lower()
                if not path_s:
                    continue
                if cls not in ("yes", "no"):
                    path_s_str = str(path_s).replace("\\", "/").lower()
                    if "/yes/" in path_s_str or path_s_str.startswith("yes/"):
                        cls = "yes"
                    elif "/no/" in path_s_str or path_s_str.startswith("no/"):
                        cls = "no"
                    else:
                        continue
                r = resolve_one(cls, str(path_s))
                if r and _is_image(r):
                    out[cls].append(r)
            else:
                s = str(item).replace("\\", "/")
                for cls in ("yes", "no"):
                    r = resolve_one(cls, s)
                    if r and _is_image(r):
                        out[cls].append(r)
                        break
        return out if (out["yes"] or out["no"]) else None

    return None


def image_path_for_json(repo_root: Path, abs_path: Path) -> str:
    try:
        rel = abs_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = abs_path
    return rel.as_posix()


def main() -> None:
    if not EXPORTS_DIR.is_dir():
        print(f"Error: missing exports dir: {EXPORTS_DIR}", file=sys.stderr)
        sys.exit(1)

    label_to_q = load_label_to_question(cot_jsonl)
    rows: list[dict] = []

    category_dirs = sorted(
        p for p in EXPORTS_DIR.iterdir()
        if p.is_dir() and category_label_from_dirname(p.name)
    )

    for cat_dir in category_dirs:
        label = category_label_from_dirname(cat_dir.name)
        assert label is not None
        manifest_path = cat_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            print(f"Skip (no manifest): {cat_dir.name}")
            continue

        buckets = parse_manifest(manifest_path, cat_dir)
        if not buckets:
            print(f"Skip (empty manifest): {cat_dir.name}")
            continue

        question = label_to_q.get(label)
        if not question:
            print(
                f"Warning: no question for label {label!r} in cot file; skipping {cat_dir.name}",
                file=sys.stderr,
            )
            continue

        for cls in ("yes", "no"):
            for img_path in sorted(buckets.get(cls, []), key=lambda p: p.as_posix()):
                if not _is_image(img_path):
                    continue
                rows.append(
                    {
                        "image": image_path_for_json(root, img_path),
                        "label": label,
                        "question": question,
                        "answer": cls,
                        "cot": "",
                    }
                )

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} lines to {OUT_JSONL}")


if __name__ == "__main__":
    main()
