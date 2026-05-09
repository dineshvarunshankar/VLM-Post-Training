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

## Commands

### Training

#### Cosmos Reason 2
```bash
python train/cosmos_reason_2/finetune_cosmosreason2_SFT.py
python train/cosmos_reason_2/finetune_cosmosreason2_SFT_no_cot.py
python train/cosmos_reason_2/finetune_cosmosreason2_GRPO.py
```

#### Gemma 4
```bash
python train/gemma4/finetune_gemma4_SFT.py
```

### Evaluation

1. **Build Test Dataset**
   ```bash
   python data/test/build_test_jsonl.py
   ```

2. **Run Inference**
   
   **Note**: For each model, you must run the **_sft** inference script first. This script initializes the predictions JSON file; subsequent scripts will append their results.

   **Cosmos Reason 2**
   ```bash
   python inference/cosmos_reason_2/inference_cosmos_sft.py
   python inference/cosmos_reason_2/inference_cosmos_base.py
   python inference/cosmos_reason_2/inference_cosmos_no_cot.py
   python inference/cosmos_reason_2/inference_cosmos_rl.py
   ```

   **Gemma 4**
   ```bash
   python inference/gemma4/inference_gemma4_sft.py
   python inference/gemma4/inference_gemma4_base.py
   ```

3. **Generate Results**
   ```bash
   python data/test/build_results_md.py
   ```

4. **Compare Outputs**
   ```bash
   python plots/compare_model_outputs.py
   ```

## Slurm (PSC Cluster)

```bash
sbatch sbatch_psc/run_cosmos.sbatch
sbatch sbatch_psc/run_gemma4.sbatch
```
