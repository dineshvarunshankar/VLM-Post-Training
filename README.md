# VLM post-training

Cosmos Reason 2 and Gemma 4 training and eval for triage yes/no tasks.

## Environment (uv)

Requires **Python ≥ 3.13** ([`pyproject.toml`](pyproject.toml)).

Install **uv** on Linux: see [Installation](https://docs.astral.sh/uv/getting-started/installation/) (`curl`/`wget` one-liners are on that page).

From this directory:

```bash
uv sync
source .venv/bin/activate   # POSIX bash/zsh; use your shell’s equivalent if different
```

[`uv.lock`](uv.lock) pins dependency versions. Re-run `uv sync` after lockfile updates.

On clusters, use whatever Python env your site provides; you only need matching packages, not necessarily uv.

## Project structure

```
VLM-Post-Training/
├── data/
│   ├── data_processor.py              # turn raw JSONL into model-ready JSONL → outputs/formatted_dataset/
│   ├── train/
│   │   └── <split>/
│   │       ├── cot_annotations/           # sft.jsonl, rlvr.jsonl, per-label *.json dirs
│   │       └── exports/
│   │           ├── bbox_map.json
│   │           └── <label>-data-vf/
│   │               ├── selection_manifest.json
│   │               └── yes|no/            # images per class
│   └── test/
│       └── <split>/
│           └── exports/
│               ├── <label>-data-vf/
│               │   ├── selection_manifest.json
│               │   └── yes|no/
│               ├── test.jsonl                 # build_test_jsonl or hand-built
│               ├── cosmos_predictions.json, gemma4_predictions.json
│               └── results.md                 # optional; build_results_md.py
├── train/          # finetune scripts (Cosmos, Gemma)
├── inference/
├── outputs/        # checkpoints, LoRA, fused weights, formatted_dataset
├── plots/
├── sbatch_psc/
├── pyproject.toml
└── uv.lock
```

Set `train_split` / `test_split` in scripts to match your `<split>` folders.

## Finetuning

```bash
python train/cosmos_reason_2/finetune_cosmosreason2_SFT.py
python train/gemma4/finetune_gemma4_SFT.py
```

Set `train_split` in the script you run. GRPO reads `rlvr.jsonl`; other paths use `sft.jsonl`.

## Eval and plots

```bash
python inference/cosmos_reason_2/inference_cosmos_sft.py
python plots/compare_model_outputs.py
```

Set `test_split` in each inference script. Optional: `python data/test/build_test_jsonl.py`, `python data/test/build_results_md.py`.

## Slurm

[`sbatch_psc/`](sbatch_psc/) scripts expect to run where you submit from (see `#SBATCH` and `cd` in each file); use conda/env activation inside the batch file as on your cluster.

```bash
sbatch sbatch_psc/run_gemma4.sbatch
```
