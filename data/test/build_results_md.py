#!/usr/bin/env python3
"""
Write data/test/<your_test_split>/exports/results.md (images + Gemma + Cosmos model fields).

Edit test_split below. Run from repo root:
  PYTHONPATH=. python data/test/build_results_md.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
test_split = "ood_80_80"

EXPORTS_DIR = root / "data" / "test" / test_split / "exports"
TEST_JSONL = EXPORTS_DIR / "test.jsonl"
GEMMA_JSON = EXPORTS_DIR / "gemma4_predictions.json"
COSMOS_JSON = EXPORTS_DIR / "cosmos_predictions.json"
OUT_MD = EXPORTS_DIR / "results.md"

F4 = "````"
COSMOS_MODEL_KEYS = ("sft_cot", "base_cot", "sft_no_cot", "rl_cot")


def _href_to_image(exports_dir: Path, repo_root: Path, image: str) -> str:
    """Path for `![](...)` relative to results.md (under exports/). `image` is from test.jsonl."""
    exports_dir = exports_dir.resolve()
    repo_root = repo_root.resolve()
    rel = Path(image.replace("\\", "/"))

    def try_path(f: Path) -> str | None:
        f = f.resolve()
        if not f.is_file():
            return None
        try:
            return f.relative_to(exports_dir).as_posix()
        except ValueError:
            return None

    if rel.parts and rel.parts[0] == "testing_exports" and len(rel.parts) > 1:
        if (h := try_path(exports_dir.joinpath(*rel.parts[1:]))):
            return h
    for base in (repo_root, exports_dir):
        if (h := try_path(base / rel)):
            return h
    if rel.name:
        hits = [p for p in exports_dir.rglob(rel.name) if p.is_file()]
        if len(hits) == 1:
            return hits[0].relative_to(exports_dir).as_posix()
    return rel.name


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def block_body(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).replace("\r\n", "\n")


def emit_block(lines: list[str], heading: str, body: str | None) -> None:
    lines.append(heading)
    lines.append("")
    lines.append(F4)
    lines.append(block_body(body))
    lines.append(F4)
    lines.append("")


def main() -> None:
    if not TEST_JSONL.is_file():
        print(f"Missing {TEST_JSONL}", file=sys.stderr)
        sys.exit(1)

    test_rows = read_jsonl(TEST_JSONL)
    gemma_rows = read_json(GEMMA_JSON)
    cosmos_rows = read_json(COSMOS_JSON)

    exports_dir = EXPORTS_DIR.resolve()

    lines: list[str] = []
    lines.append("# Eval results gallery")
    lines.append("")
    lines.append(f"- Split: `{test_split}`")
    lines.append("")
    lines.append("## Samples")
    lines.append("")

    titles = {
        "sft_cot": "##### SFT + CoT (`sft_cot`)",
        "base_cot": "##### Base + CoT (`base_cot`)",
        "sft_no_cot": "##### SFT no-CoT (`sft_no_cot`)",
        "rl_cot": "##### RL + CoT (`rl_cot`)",
    }

    for i, tr in enumerate(test_rows):
        img = tr.get("image", "")
        href = _href_to_image(exports_dir, root, img)

        label = tr.get("label", "")
        lines.append(f"### {i + 1}. `{label}`")
        lines.append("")
        lines.append(f"![sample {i + 1}](./{href})")
        lines.append("")
        lines.append(f"- **image:** `{href}`")
        lines.append(f"- **question:** {tr.get('question', '')}")
        lines.append(f"- **gt_answer:** `{tr.get('answer', '')}`")
        lines.append("")

        if i < len(gemma_rows):
            g = gemma_rows[i]
            lines.append("#### Gemma")
            lines.append("")
            emit_block(lines, "##### `prediction`", g.get("prediction"))
            base_out = g.get("base")
            if base_out is not None and str(base_out).strip():
                emit_block(lines, "##### Base (`base`)", base_out)
        else:
            lines.append("#### Gemma\n\n*(no row in gemma4_predictions.json)*\n")

        if i < len(cosmos_rows):
            c = cosmos_rows[i]
            lines.append("#### Cosmos (model outputs)")
            lines.append("")
            emitted = set()
            for key in COSMOS_MODEL_KEYS:
                if key not in c or not str(c.get(key, "")).strip():
                    continue
                emitted.add(key)
                emit_block(lines, titles.get(key, f"##### `{key}`"), c.get(key))
            for key in sorted(c.keys()):
                if key in ("image", "question", "gt_answer", "gt_cot") or key in emitted:
                    continue
                if key in COSMOS_MODEL_KEYS:
                    continue
                emit_block(lines, f"##### Other `{key}`", c.get(key))
        else:
            lines.append("#### Cosmos\n\n*(no row in cosmos_predictions.json)*\n")

        lines.append("---")
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
