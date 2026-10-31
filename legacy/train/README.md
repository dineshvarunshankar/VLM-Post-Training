# Legacy training

[Legacy overview](../README.md) · [Current video training](../../scripts/train/README.md)

These scripts use Unsloth and image-based JSONL data.
They do not consume the current per-clip video datasets directly.

## Available scripts

Paths are relative to `legacy/train/`.

| Script | Purpose |
|---|---|
| `gemma4/finetune_gemma4_12B_SFT.py` | Gemma 4 12B base SFT |
| `gemma4/finetune_gemma4_12B_it_SFT.py` | Gemma 4 12B instruction-model SFT |
| `gemma4/finetune_gemma4_31B_SFT.py` | Gemma 4 31B SFT |
| `cosmos_reason_2/finetune_cosmosreason2_SFT.py` | Cosmos Reason2 SFT with reasoning |
| `cosmos_reason_2/finetune_cosmosreason2_SFT_no_cot.py` | Cosmos Reason2 answer-only SFT |
| `cosmos_reason_2/finetune_cosmosreason2_GRPO.py` | Cosmos Reason2 GRPO |
| `cosmos_reason_2/reward_functions.py` | GRPO reward functions |

## Prepare inputs

Install the [legacy environment](../README.md#environment).
Set `train_split`, the model path, training configuration, and output paths inside the selected script.
Several scripts still contain placeholders.

Expected files:

```text
legacy/data/train/<split>/cot_annotations/sft.jsonl
legacy/data/train/<split>/cot_annotations/rlvr.jsonl
```

Rows contain `image`, `question`, `cot`, and `answer`. The GRPO conversion also reads `subject_bbox`.
Image paths are relative to `legacy/`, such as `data/train/<split>/exports/.../file.jpg`.
The converter copies image paths unchanged and writes chat records under `outputs/formatted_dataset/`.

## Credentials

Export credentials in the shell before running a script:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Authentication for restricted model downloads |
| `WANDB_API_KEY` | Online W&B tracking |
| `WANDB_MODE` | Optional `offline` or `disabled` mode |

The scripts assign `WANDB_PROJECT` internally. External values do not override that assignment.
They do not read the current pipeline's dataset or Gemini variables.

## Run

After reviewing the script, run from `legacy/`:

```bash
uv run --python 3.13 python -m train.gemma4.finetune_gemma4_12B_SFT
```

This starts training, not an inspection. Use the corresponding module name for another script.
Checkpoints, adapters, tokenizer files, and merged weights use the output paths configured inside that script.

Conversion from current video annotations requires a separate reviewed data adapter.
The current [answer-only video trainers](../../scripts/train/README.md) already provide the standard-library workflow.
