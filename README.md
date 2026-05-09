# VLM post-training

Train and evaluate vision-language models (Cosmos Reason 2, Gemma 4) on triage-style yes/no questions.

## Directory layout

| Path | Purpose |
|------|--------|
| `data/train/<split>/cot_annotations/` | Raw **`sft.jsonl`** and **`rlvr.jsonl`** for fine-tuning |
| `data/test/<split>/exports/` | Eval images (`*-data-vf/yes|no/`), **`selection_manifest.json`** per category, **`test.jsonl`**, **`cosmos_predictions.json`**, **`gemma4_predictions.json`**, **`results.md`** |
| `outputs/formatted_dataset/` | HF-style JSONL from **`data/data_processor.py`** (under ignored **`outputs/`**) |
| `plots/runs/<YYYYMMDD_HHMMSS>/` | Default output for **`plots/compare_model_outputs.py`** |

Scripts define **`train_split`** / **`test_split`** (lowercase) with placeholder string values **`your_train_split`** / **`your_test_split`** — replace those strings with your split ids.

## Imports and `PYTHONPATH`

Run from the **repository root** with the repo on the import path, for example:

```bash
cd /path/to/VLM-Post-Training
PYTHONPATH=. python train/cosmos_reason_2/finetune_cosmosreason2_SFT.py
```

Finetune scripts use **`from data.data_processor import DataProcessor`** with no `sys.path` hacks.

## Scripts under `data/test/`

```bash
PYTHONPATH=. python data/test/build_test_jsonl.py    # writes data/test/<split>/exports/test.jsonl
PYTHONPATH=. python data/test/build_results_md.py  # writes data/test/<split>/exports/results.md
```

Edit **`train_split`**, **`test_split`**, and **`cot_jsonl`** (in `build_test_jsonl`) at the top of each script.

## Training

Set **`train_split = "your_train_split"`** (replace **`your_train_split`** with your split id) in each finetune script. GRPO reads **`rlvr.jsonl`**; other SFT paths use **`sft.jsonl`**.

## Inference

Each file under **`inference/`** sets **`test_split = "your_test_split"`** and builds paths with `f"data/test/{test_split}/exports/..."`.

## Plots

```bash
PYTHONPATH=. python plots/compare_model_outputs.py
```

**`plots/compare_model_outputs.py`** uses **`test_split`** for default Cosmos/Gemma JSON paths; override with **`--cosmos-json`**, **`--gemma-json`**, **`--outdir`**.

## Slurm

Submit from the **repository root** (so **`SLURM_SUBMIT_DIR`** is that directory):

```bash
cd /path/to/VLM-Post-Training
sbatch sbatch_psc/run_gemma4.sbatch
```

Each sbatch **`cd "$SLURM_SUBMIT_DIR"`**, sets **`PYTHONPATH`**, then runs **`python inference/...`**.
