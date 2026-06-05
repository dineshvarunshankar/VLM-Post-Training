# VLM Post-Training

Post-training pipeline for Visual Language Models (VLMs).

## Setup

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup Environment
```bash
uv sync
source .venv/bin/activate
```

## Project structure

```text
VLM-Post-Training/
├── data/
│   ├── data_processor.py              # raw JSONL → outputs/formatted_dataset/ (chat messages)
│   ├── train/
│   │   └── <split>/
│   │       ├── cot_annotations/           # sft.jsonl, rlvr.jsonl
│   │       └── exports/
│   │           └── <label>-data-vf/
│   │               ├── selection_manifest.json
│   │               └── yes|no/
│   └── test/
│       └── <split>/
│           └── exports/
│               ├── test.jsonl
│               ├── cosmos_predictions.json, gemma4_predictions.json
│               └── results.md
├── train/          # finetune scripts (Cosmos, Gemma)
├── inference/
├── outputs/        # checkpoints, LoRA, fused weights, formatted_dataset
├── plots/
├── sbatch_psc/
├── pyproject.toml
└── uv.lock
```

Set `train_split` / `test_split` in each script to match your `data/train/<split>/` and `data/test/<split>/` folders.

### Image paths in JSONL

Each row’s `image` must be a **repo-root-relative** path (e.g. `data/train/100_100/exports/.../file.jpg`), not `exports/...` alone. Set this when you create `sft.jsonl` / `rlvr.jsonl`. `data/test/build_test_jsonl.py` writes full paths for test data.

`data_processor.py` only converts rows to chat `messages` format; it copies `image` unchanged.

### Training

#### Cosmos Reason 2
```bash
PYTHONPATH=. python train/cosmos_reason_2/finetune_cosmosreason2_SFT.py
PYTHONPATH=. python train/cosmos_reason_2/finetune_cosmosreason2_SFT_no_cot.py
PYTHONPATH=. python train/cosmos_reason_2/finetune_cosmosreason2_GRPO.py
```

#### Gemma 4
```bash
# 12B base (unsloth/gemma-4-12b)
PYTHONPATH=. python train/gemma4/finetune_gemma4_12B_SFT.py

# 12B instruct (unsloth/gemma-4-12b-it) — use for CoT SFT
PYTHONPATH=. python train/gemma4/finetune_gemma4_12B_it_SFT.py

# 31B instruct
PYTHONPATH=. python train/gemma4/finetune_gemma4_31B_SFT.py
```

### Evaluation

1. **Build test JSONL** — edit `train_split` / `test_split` in `data/test/build_test_jsonl.py`, then:
   ```bash
   PYTHONPATH=. python data/test/build_test_jsonl.py
   ```

2. **Run inference**

   Run the `*_sft` script first; it creates the predictions JSON. Later scripts append fields to the same file.

   **Cosmos Reason 2**
   ```bash
   PYTHONPATH=. python inference/cosmos_reason_2/inference_cosmos_sft.py
   PYTHONPATH=. python inference/cosmos_reason_2/inference_cosmos_base.py
   PYTHONPATH=. python inference/cosmos_reason_2/inference_cosmos_no_cot.py
   PYTHONPATH=. python inference/cosmos_reason_2/inference_cosmos_rl.py
   ```

   **Gemma 4**
   ```bash
   PYTHONPATH=. python inference/gemma4/inference_gemma4_12B_SFT.py
   PYTHONPATH=. python inference/gemma4/inference_gemma4_31B_sft.py
   PYTHONPATH=. python inference/gemma4/inference_gemma4_base.py
   ```

3. **Generate results** — edit `test_split` in `data/test/build_results_md.py`, then:
   ```bash
   PYTHONPATH=. python data/test/build_results_md.py
   ```

4. **Compare outputs**
   ```bash
   PYTHONPATH=. python plots/compare_model_outputs.py
   ```

## Slurm (PSC)

Submit from the repo root (`VLM-Post-Training/`), not from `sbatch_psc/`:

```bash
cd /path/to/VLM-Post-Training
uv sync
sbatch sbatch_psc/run_cosmos.sbatch          # Cosmos inference
sbatch sbatch_psc/run_gemma4_12B.sbatch      # Gemma 4 12B base training
sbatch sbatch_psc/run_gemma4_12B_it.sbatch   # Gemma 4 12B instruct training
```
