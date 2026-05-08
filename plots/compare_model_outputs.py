import argparse
import json
import os
import re
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|therefore|hence|thus)\s*(?:is|:)?\s*\b(yes|no)\b",
    re.IGNORECASE,
)
GEMMA_MARKDOWN_ANSWER_RE = re.compile(
    r"(?:\*\*\s*)?(?:Final\s+)?Answer\s*(?:\*\*\s*)?:\s*(?:/\s*)?\b(yes|no)\b",
    re.IGNORECASE | re.MULTILINE,
)

COSMOS_RL_LABEL = "Cosmos RL"
COSMOS_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Cosmos Base": ["base_cot", "base", "cosmos_base"],
    "Cosmos SFT": ["sft_cot", "prediction_cot", "prediction", "cosmos_sft"],
    COSMOS_RL_LABEL: ["rl_cot", "rl_prediction", "grpo_prediction", "prediction_rl", "cosmos_rl"],
    "Cosmos SFT no-CoT": ["sft_no_cot", "prediction_no_cot", "no_cot_prediction", "cosmos_no_cot"],
}
GEMMA_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "Gemma-4-31B Base": ["base", "base_cot", "gemma_base"],
    "Gemma-4-31B SFT": ["prediction", "prediction_cot", "gemma_sft"],
}
GEMMA_JSON_CANDIDATES = (
    "test/gemma4_predictions.json",
    "testing_exports/gemma4_predictions.json",
)


def read_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_yes_no(text: str) -> str:
    if text is None:
        return ""
    raw = str(text)
    tag = ANSWER_TAG_RE.search(raw)
    if tag:
        tag_text = tag.group(1)
        match = YES_NO_RE.search(tag_text)
        return match.group(1).lower() if match else ""
    md_matches = GEMMA_MARKDOWN_ANSWER_RE.findall(raw)
    if md_matches:
        return md_matches[-1].lower()
    final_matches = FINAL_ANSWER_RE.findall(raw)
    if final_matches:
        return final_matches[-1].lower()
    all_matches = YES_NO_RE.findall(raw)
    return all_matches[-1].lower() if all_matches else ""


def f1_binary_yes(y_true: List[str], y_pred: List[str]) -> float:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == "yes" and t == "yes":
            tp += 1
        elif p == "yes" and t != "yes":
            fp += 1
        elif p != "yes" and t == "yes":
            fn += 1
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def first_existing_key(rows: List[dict], candidates: List[str]) -> Optional[str]:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for key in candidates:
        if key in keys:
            return key
    return None


def gather_models(cosmos_rows: List[dict], gemma_rows: List[dict]) -> Dict[str, Tuple[List[dict], str]]:
    models: Dict[str, Tuple[List[dict], str]] = {}
    for label, keys in COSMOS_MODEL_CANDIDATES.items():
        key = first_existing_key(cosmos_rows, keys)
        if key:
            models[label] = (cosmos_rows, key)
    for label, keys in GEMMA_MODEL_CANDIDATES.items():
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


def confusion_counts(y_true: List[str], y_pred: List[str]) -> np.ndarray:
    mat = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in ("yes", "no") or p not in ("yes", "no"):
            continue
        i = 0 if t == "yes" else 1
        j = 0 if p == "yes" else 1
        mat[i, j] += 1
    return mat


def build_agreement_matrix(model_metrics: Dict[str, dict]) -> Tuple[List[str], np.ndarray]:
    names = list(model_metrics.keys())
    m = np.eye(len(names), dtype=float)
    for i, j in combinations(range(len(names)), 2):
        a = model_metrics[names[i]]
        b = model_metrics[names[j]]
        paired = [(pa, pb) for pa, pb in zip(a["pred"], b["pred"]) if pa in ("yes", "no") and pb in ("yes", "no")]
        if paired:
            score = sum(int(x == y) for x, y in paired) / len(paired)
        else:
            score = 0.0
        m[i, j] = m[j, i] = score
    return names, m


def save_metric_bars(model_metrics: Dict[str, dict], outdir: str) -> None:
    names = list(model_metrics.keys())
    if not names:
        return
    c_pred_yes, e_pred_yes = "#1976d2", "#0d47a1"
    c_pred_no, e_pred_no = "#d32f2f", "#b71c1c"
    spacing = 3.4
    w_gt = 0.44
    w_pr = 0.26
    gt_green = "#c8e6c9"
    gt_edge = "#81c784"
    centers = np.arange(len(names), dtype=float) * spacing
    ymax = 1
    for name in names:
        m = model_metrics[name]
        gt, pred = m["gt"], m["pred"]
        vals = (
            sum(1 for t in gt if t == "yes"),
            sum(1 for t in gt if t == "no"),
            sum(1 for p in pred if p == "yes"),
            sum(1 for p in pred if p == "no"),
        )
        ymax = max(ymax, max(vals))
    fig_w = max(10.0, 3.5 * len(names) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6.35))

    def annotate_overlay_pair(x_pos: float, gt_h: float, pr_h: float, ha: str) -> None:
        top = max(gt_h, pr_h, 1)
        gy, py = int(gt_h), int(pr_h)
        ax.text(
            x_pos,
            top + ymax * 0.04,
            f"GT $\\mathbf{{{gy}}}$\nPred $\\mathbf{{{py}}}$",
            ha=ha,
            va="bottom",
            fontsize=7,
            linespacing=1.08,
            color="#222",
        )

    for mi, name in enumerate(names):
        mdt = model_metrics[name]
        gt, pred = mdt["gt"], mdt["pred"]
        gt_yes = sum(1 for t in gt if t == "yes")
        gt_no = sum(1 for t in gt if t == "no")
        pred_yes = sum(1 for p in pred if p == "yes")
        pred_no = sum(1 for p in pred if p == "no")
        ctr = centers[mi]
        x_yes = ctr - 0.55
        x_no = ctr + 0.55
        ax.bar(x_yes, gt_yes, width=w_gt, color=gt_green, edgecolor=gt_edge, linewidth=0.7, zorder=1)
        ax.bar(x_yes, pred_yes, width=w_pr, color=c_pred_yes, edgecolor=e_pred_yes, linewidth=0.5, zorder=2)
        annotate_overlay_pair(x_yes, gt_yes, pred_yes, "right")
        ax.bar(x_no, gt_no, width=w_gt, color=gt_green, edgecolor=gt_edge, linewidth=0.7, zorder=1)
        ax.bar(x_no, pred_no, width=w_pr, color=c_pred_no, edgecolor=e_pred_no, linewidth=0.5, zorder=2)
        annotate_overlay_pair(x_no, gt_no, pred_no, "left")
    pad = max(1.0, ymax * 0.08)
    ax.set_ylim(-pad, ymax * 1.28)
    ax.set_ylabel("Count")
    ax.set_xticks(centers)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor=gt_green, edgecolor=gt_edge, linewidth=1, label="GT"),
            Patch(facecolor=c_pred_yes, edgecolor=e_pred_yes, linewidth=1, label="Pred yes"),
            Patch(facecolor=c_pred_no, edgecolor=e_pred_no, linewidth=1, label="Pred no"),
        ],
        loc="upper right",
        framealpha=0.92,
    )
    y_lbl = -pad * 0.55
    for c in centers:
        ax.text(c - 0.55, y_lbl, "yes", ha="center", va="top", fontsize=10, color="#333")
        ax.text(c + 0.55, y_lbl, "no", ha="center", va="top", fontsize=10, color="#333")
    ax.set_title("GT vs prediction marginals")
    fig.savefig(os.path.join(outdir, "comparison_core_metrics.png"), dpi=180, bbox_inches="tight")
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
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if mat[i, j] < 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comparison_agreement_heatmap.png"), dpi=180)
    plt.close(fig)


def save_confusion_grid(model_metrics: Dict[str, dict], outdir: str) -> None:
    names = list(model_metrics.keys())
    if not names:
        return
    global_max = max(
        confusion_counts(model_metrics[name]["gt"], model_metrics[name]["pred"]).max()
        for name in names
    )
    if global_max <= 0:
        global_max = 1
    fig_w = min(24.0, 3.8 * len(names) + 1.4)
    fig, axs = plt.subplots(1, len(names), figsize=(fig_w, 3.9))
    axes_flat = np.asarray(axs).reshape(-1)
    norm = Normalize(0.0, float(global_max))
    blues_cm = plt.colormaps["Blues"]
    reds_cm = plt.colormaps["Reds"]
    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        mat = confusion_counts(model_metrics[name]["gt"], model_metrics[name]["pred"])
        canvas = np.zeros((2, 2, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                v = int(mat[i, j])
                t = norm(float(v))
                cmap = blues_cm if j == 0 else reds_cm
                canvas[i, j, :] = cmap(t)
        ax.imshow(canvas, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1], labels=["Pred yes", "Pred no"])
        ax.set_yticks([0, 1], labels=["GT yes", "GT no"])
        for i in range(2):
            for j in range(2):
                v = int(mat[i, j])
                r, g, b, _a = canvas[i, j]
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if lum < 0.55 else "black",
                )
    fig.suptitle("Confusion matrices", fontsize=11, y=1.03)
    fig.subplots_adjust(right=0.98, top=0.80)
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


def resolve_gemma_path(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in GEMMA_JSON_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-json", default="testing_exports/cosmos_predictions.json")
    ap.add_argument("--outdir", default="outputs/analysis/OOD_cosmos")
    ap.add_argument("--gemma-json", default=None)
    args = ap.parse_args()
    cosmos_rows = read_json(args.cosmos_json)
    gemma_path = resolve_gemma_path(args.gemma_json)
    gemma_rows = read_json(gemma_path) if gemma_path else []
    models = gather_models(cosmos_rows, gemma_rows)
    if not models:
        raise SystemExit("No models: add prediction columns to cosmos or gemma JSON.")
    os.makedirs(args.outdir, exist_ok=True)
    metrics = {name: compute_metrics(rows, key) for name, (rows, key) in models.items()}
    save_metric_bars(metrics, args.outdir)
    save_agreement_heatmap(metrics, args.outdir)
    save_confusion_grid(metrics, args.outdir)
    print_summary(metrics)
    print(f"\nSaved charts to: {args.outdir}")


if __name__ == "__main__":
    main()
