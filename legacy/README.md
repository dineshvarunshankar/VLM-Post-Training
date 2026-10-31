# Legacy Unsloth workflow

[Current repository guide](../README.md) · [Current training](../scripts/train/README.md)

This folder preserves the previous image-based Unsloth scripts.
Use [scripts/train](../scripts/train/README.md) for the current video training workflow.
The legacy environment and scripts have not been revalidated with the new datasets.

## Environment

From the repository root:

```bash
cd legacy
uv python install 3.13
uv sync --locked --python 3.13
```

Run the commands in this guide from `legacy/`.
Review script configuration and paths before running training, inference, or cluster jobs.

## Files

| Folder | Contents |
|---|---|
| `train/` | Gemma and Cosmos training scripts |
| `data/` | Image-based JSONL conversion and evaluation helpers |
| `inference/` | Model inference scripts |
| `plots/` | Output comparisons |
| `sbatch_psc/` | Historical PSC job scripts |

Training configuration, input format, and module names are in the [legacy training guide](train/README.md).

## Evaluation

Set the split names and model paths inside the selected scripts first.
Build the test JSONL:

```bash
PYTHONPATH=. uv run --python 3.13 python data/test/build_test_jsonl.py
```

Run the matching inference script. For the historical append workflow, run its SFT inference script first:

```bash
PYTHONPATH=. uv run --python 3.13 python inference/gemma4/inference_gemma4_12B_SFT.py
```

Build the report and compare outputs:

```bash
PYTHONPATH=. uv run --python 3.13 python data/test/build_results_md.py
PYTHONPATH=. uv run --python 3.13 python plots/compare_model_outputs.py
```

The Slurm files retain historical paths, allocations, and launch commands.
Review them before submission. They are not the current cluster workflow.
