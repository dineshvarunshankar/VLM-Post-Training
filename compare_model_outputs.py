import json
import os
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# Set paths here directly.
COSMOS_JSON = "test/cosmos_predictions.json"
GEMMA_JSON = "test/gemma4_predictions.json"
OUTDIR = "outputs/analysis/model_comparison"


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_yes_no(text: str) -> str:
    if text is None:
        return ""
    raw = str(text)
    tag = ANSWER_TAG_RE.search(raw)
    if tag:
        raw = tag.group(1)
    match = YES_NO_RE.search(raw)
    return match.group(1).lower() if match else ""


def f1_binary_yes(y_true: List[str], y_pred: List[str]) -> float:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == "yes" and t == "yes":
            tp += 1
        elif p == "yes" and t != "yes":
            fp += 1
        elif p != "yes" and t == "yes":
            fn += 1
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom else 0.0


def confusion_counts(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    # rows = gt [yes, no], cols = pred [yes, no]
    mat = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in ("yes", "no") or p not in ("yes", "no"):
            continue
        i = 0 if t == "yes" else 1
        j = 0 if p == "yes" else 1
        mat[i, j] += 1
    return mat


def first_existing_key(rows: List[dict], candidates: List[str]) -> Optional[str]:
    if not rows:
        return None
    row_keys = set(rows[0].keys())
    for key in candidates:
        if key in row_keys:
            return key
    return None


def gather_models(cosmos_rows: List[dict], gemma_rows: List[dict]) -> Dict[str, Tuple[List[dict], str]]:
    cosmos_candidates = {
        "Cosmos Base": ["base_cot", "base", "cosmos_base"],
        "Cosmos SFT": ["prediction_cot", "prediction", "cosmos_sft"],
        "Cosmos RL": ["rl_prediction", "grpo_prediction", "prediction_rl", "cosmos_rl"],
        "Cosmos SFT no-CoT": ["no_cot_prediction", "prediction_no_cot", "cosmos_no_cot"],
    }
    gemma_candidates = {
        "Gemma-4-31B Base": ["base", "base_cot", "gemma_base"],
        "Gemma-4-31B SFT": ["prediction", "prediction_cot", "gemma_sft"],
    }

    models: Dict[str, Tuple[List[dict], str]] = {}

    for label, keys in cosmos_candidates.items():
        key = first_existing_key(cosmos_rows, keys)
        if key:
            models[label] = (cosmos_rows, key)

    for label, keys in gemma_candidates.items():
        key = first_existing_key(gemma_rows, keys)
        if key:
            models[label] = (gemma_rows, key)

    return models


def compute_metrics(rows: List[dict], pred_key: str) -> dict:
    gt = [parse_yes_no(r.get("gt_answer", r.get("answer", ""))) for r in rows]
    pred_raw = [r.get(pred_key, "") for r in rows]
    pred = [parse_yes_no(v) for v in pred_raw]
    valid_mask = [p in ("yes", "no") and t in ("yes", "no") for p, t in zip(pred, gt)]

    valid_gt = [t for t, m in zip(gt, valid_mask) if m]
    valid_pred = [p for p, m in zip(pred, valid_mask) if m]

    n = len(rows)
    n_valid = len(valid_pred)
    acc = (sum(int(a == b) for a, b in zip(valid_gt, valid_pred)) / n_valid) if n_valid else 0.0
    f1 = f1_binary_yes(valid_gt, valid_pred) if n_valid else 0.0
    yes_rate = (sum(1 for p in valid_pred if p == "yes") / n_valid) if n_valid else 0.0

    return {
        "n": n,
        "valid_rate": (n_valid / n) if n else 0.0,
        "accuracy": acc,
        "f1_yes": f1,
        "yes_rate": yes_rate,
        "gt": gt,
        "pred": pred,
    }


def build_agreement_matrix(model_metrics: Dict[str, dict]) -> Tuple[List[str], np.ndarray]:
    names = list(model_metrics.keys())
    m = np.eye(len(names), dtype=float)

    for i, j in combinations(range(len(names)), 2):
        a = model_metrics[names[i]]
        b = model_metrics[names[j]]
        paired = [(pa, pb) for pa, pb in zip(a["pred"], b["pred"]) if pa in ("yes", "no") and pb in ("yes", "no")]
        if not paired:
            score = 0.0
        else:
            score = sum(int(x == y) for x, y in paired) / len(paired)
        m[i, j] = m[j, i] = score
    return names, m


def save_metric_bars(model_metrics: Dict[str, dict], outdir: str) -> None:
    names = list(model_metrics.keys())
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width, [model_metrics[n]["accuracy"] for n in names], width=width, label="Accuracy")
    ax.bar(x, [model_metrics[n]["f1_yes"] for n in names], width=width, label="F1 (yes)")
    ax.bar(x + width, [model_metrics[n]["valid_rate"] for n in names], width=width, label="Valid prediction rate")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Core Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    # Show dataset GT distribution on the same chart.
    first = model_metrics[names[0]]
    gt_yes = int(sum(1 for v in first["gt"] if v == "yes"))
    gt_no = int(sum(1 for v in first["gt"] if v == "no"))
    ax.text(
        0.99,
        0.98,
        f"GT: yes={gt_yes}, no={gt_no}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_core_metrics.png"), dpi=180)
    plt.close(fig)


def save_agreement_heatmap(model_metrics: Dict[str, dict], outdir: str) -> None:
    names, mat = build_agreement_matrix(model_metrics)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_title("Pairwise Prediction Agreement")

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] < 0.5 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_agreement_heatmap.png"), dpi=180)
    plt.close(fig)


def save_confusion_grid(model_metrics: Dict[str, dict], outdir: str) -> None:
    names = list(model_metrics.keys())
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, name in enumerate(names):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        m = confusion_counts(model_metrics[name]["gt"], model_metrics[name]["pred"])
        gt_yes = int(sum(1 for v in model_metrics[name]["gt"] if v == "yes"))
        gt_no = int(sum(1 for v in model_metrics[name]["gt"] if v == "no"))
        im = ax.imshow(m, cmap="Blues")
        ax.set_title(f"{name}\nGT yes={gt_yes}, GT no={gt_no}")
        ax.set_xticks([0, 1], labels=["Pred yes", "Pred no"])
        ax.set_yticks([0, 1], labels=["GT yes", "GT no"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center")
        # Row totals show how many GT yes/no samples were counted in matrix.
        ax.text(1.7, 0, f"row total={m[0, 0] + m[0, 1]}", va="center", fontsize=9)
        ax.text(1.7, 1, f"row total={m[1, 0] + m[1, 1]}", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(len(names), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle("Confusion Matrices by Model", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_confusion_matrices.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_summary(model_metrics: Dict[str, dict]) -> None:
    print("\n=== Comparison Summary ===")
    for name, m in model_metrics.items():
        print(
            f"{name:20s} "
            f"acc={m['accuracy']:.4f} "
            f"f1_yes={m['f1_yes']:.4f} "
            f"valid={m['valid_rate']:.4f} "
            f"yes_rate={m['yes_rate']:.4f}"
        )


def main() -> None:
    cosmos_rows = read_json(COSMOS_JSON)
    gemma_rows = read_json(GEMMA_JSON)

    models = gather_models(cosmos_rows, gemma_rows)
    if not models:
        raise ValueError("No expected prediction columns found in the provided JSON files.")

    os.makedirs(OUTDIR, exist_ok=True)

    metrics = {}
    for name, (rows, key) in models.items():
        metrics[name] = compute_metrics(rows, key)

    save_metric_bars(metrics, OUTDIR)
    save_agreement_heatmap(metrics, OUTDIR)
    save_confusion_grid(metrics, OUTDIR)
    print_summary(metrics)
    print(f"\nSaved charts to: {OUTDIR}")


if __name__ == "__main__":
    main()
