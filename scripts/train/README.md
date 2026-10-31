# Answer-only training

[Repository guide](../../README.md) · [Preprocessing](../preprocess/README.md) · [CoT generation](../cot/README.md)

Train RGB video models from human answers with TRL, Transformers, and PEFT LoRA.
Medical VQA targets contain category JSON with integer `0`/`1` values. Thinking is disabled.
CoT annotations are not required for these recipes.

## 1. Choose a run YAML

Use one complete YAML per run. Copy an example within the same directory:

```bash
cp scripts/train/configs/runs/qwen9b_stabilized_medical.yaml scripts/train/configs/runs/my_run.yaml
```

| Example | Model | Videos | Tasks |
|---|---|---|---|
| [qwen9b_stabilized_medical.yaml](configs/runs/qwen9b_stabilized_medical.yaml) | Qwen 3.5 9B | Stabilized | Medical VQA |
| [gemma12b_combined_auxiliary.yaml](configs/runs/gemma12b_combined_auxiliary.yaml) | Gemma 4 12B | Native + stabilized | Medical VQA + auxiliary |

Both examples start with SFT. Larger model profiles are available in [configs/models.yaml](configs/models.yaml).
These profiles still require full-model GPU checks.

Set the run YAML fields:

| Field | What to set |
|---|---|
| `run.name`, `run.date` | Unique name and quoted `YYYY-MM-DD` date |
| `recipe` | `sft`, `rl_only`, or `sft_then_rl` |
| `model`, `model_settings` | Model identifier and its complete pinned profile |
| `paths.media_root` | Local parent of `06` and `07` |
| `paths.split_dir` | Directory containing `train.jsonl`, `validation.jsonl`, and `test.jsonl` |
| `paths.weights_root` | Local parent of `<model-repository>/<revision>` |
| `paths.output_root` | Local parent for dated training runs |
| `paths.catalog`, `paths.prompts` | RGB catalog and medical prompt YAML |
| `dataset.sources` | `[stabilized]` or `[native, stabilized]` |
| `mixture` | Enabled tasks and sampling |
| `sft`, `rl` | Training configuration for each stage |
| `execution` | Training GPU count, device IDs, and external vLLM endpoint/name |
| `cloud` | Base-weight source and fine-tuned destination |
| `wandb` | Entity, project, mode, and tracking configuration |
| `env_file` | Credentials file. Shell values take precedence |

Relative paths resolve from the YAML directory. The example paths are placeholders.
When changing models, copy the matching full profile from `configs/models.yaml` into `model_settings`.

Keep credentials in `.env` or the shell. Keep paths and run choices in the run YAML.
The root environment template also supports preprocessing and older commands. Those variables are not required by the run-YAML launcher.

| Recipe | Starting weights | Stages |
|---|---|---|
| `sft` | Verified cloud base weights | SFT |
| `rl_only` | Verified cloud base weights | GSPO with a new LoRA adapter |
| `sft_then_rl` | Verified cloud base weights | SFT, then GSPO from the best SFT adapter |

For `sft_then_rl`, leave `sft_run: null` to run both stages.
To reuse a completed SFT run, set `sft_run` to its directory. The launcher checks compatibility before starting RL.

## 2. Prepare the dataset

Complete [preprocessing](../preprocess/README.md) first. Both datasets use this layout:

```text
<dataset-root>/
├── 00-data_website/medical/completed_clips.csv
├── 06-native_media_dataset/<clip_id>/
│   ├── <clip_id>.mp4
│   ├── <clip_id>.json
│   ├── all_annotations/
│   ├── provenance.json
│   └── GT/medical_vqa.json
├── 07-stabilized_media_dataset/<clip_id>/
└── 08-train_val_test/<split-date>/answer_only/
```

GT and split preparation use the current completion CSV, not exported completion flags.
Use `--completed-manifest /path/to/completed_clips.csv` when the CSV is elsewhere.

### Human-answer targets

Preview target extraction:

```bash
uv run python scripts/train/prepare.py gt \
  --media-root /path/to/dataset-root \
  --catalog scripts/train/configs/catalogs/RGB/label_catalog.yaml --inspect
```

Remove `--inspect` to create or refresh `GT/medical_vqa.json`.
Human `true`/`false` values become integer `1`/`0` values. Source annotations remain unchanged.
Missing answers, incomplete categories, and rule conflicts are reported. Missing labels are never filled with zeros.
Review the exclusions before training.

GT extraction requires `--catalog`. The example uses [the RGB catalog](configs/catalogs/RGB/label_catalog.yaml). Keep definitions, rules, and prompts synchronized with deployment.
Review human answers after changing label meanings or selection rules.

### Train, validation, and test splits

Preview a split:

```bash
uv run python scripts/train/create_splits.py \
  --media-root /path/to/dataset-root --modality RGB \
  --percentage 5 --date 2026-10-31 --inspect
```

Choose your split date. Remove `--inspect` to write:

```text
08-train_val_test/<split-date>/answer_only/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── summary.json
```

`--percentage 5` means 5% validation and 5% test, measured in native/stabilized clip pairs.
The default is 5. Interactive runs ask for omitted percentages and modality. Noninteractive runs require `--modality`.
Existing split directories are never overwritten. `--split-root` changes the parent directory. The default seed is 42.

The selector:

- Keeps native and stabilized views together.
- Prioritizes observable positives while retaining assessed-absent and not-assessable coverage.
- Includes at most one segment per held-out casualty group.
- Excludes other segments of that held-out casualty from all splits.
- Keeps source recordings separate across splits and seeks year and recording diversity.

Grouping can change the achieved percentages. `summary.json` reports actual sizes, coverage, exclusions, and fingerprints.
Person IDs group casualties within their source recording. They do not identify the same person across different recordings.
Validation and test are deliberately positive-enriched, so their scores do not estimate natural label prevalence.

Each JSONL row references one medical question:

```json
{"variant":"native","clip":"97g7c6735tdk","category":"severe_hemorrhage"}
```

The split files do not duplicate media or answers. Training loads answers from each clip's `GT/medical_vqa.json`.
Split creation currently supports answer-only data. A future `cot/` directory can share the dated parent.
IR split preparation requires an explicit thermal catalog. The current training recipes remain RGB-only.

### Stage data on the training machine

Finish media and GT preparation before copying a fixed dataset snapshot.
Copy the complete referenced clip folders, preserving the `06` and `07` directory names.
Copy the split directory and use the same catalog and prompts.
Set `paths.media_root` and `paths.split_dir` to the destination paths.

Training does not need `04`, `05`, or a live connection to the annotation site.
Retain the completion CSV if you plan to repeat GT or split preparation on that machine.
Local media storage avoids repeated network reads during decoding. Keep checkpoints and exports on persistent storage.
Data transfer and cluster submission remain separate from the training scripts.

## 3. Download base weights

Run from the repository root after the [base installation](../../README.md#installation).
Select the model and the same weights directory used in your run YAML:

```bash
uv run python scripts/train/model_storage.py download \
  --model qwen3.5-9b --weights-root /path/to/model-weights/weights --inspect
```

Remove `--inspect` to download. This command uses the matching profile in `configs/models.yaml`.
Keep its revision consistent with `model_settings` in the run YAML.
For a custom revision, supply `--models /path/to/models.yaml` with the same profile.

The default source is `vlm-zoo/weights/<model-repository>/<revision>/`, with fallback to the older repository-only prefix.
Downloads check revisions and checksums, resume interrupted transfers, and reject different existing content.
Training uses the verified local copy with network fallback disabled. Public base-weight downloads do not require an HF token.

The standalone downloader reads cloud defaults from `configs/storage.yaml`. It does not read the run YAML.
For a different source, set `AIRLAB_SOURCE_CONTAINER` and `AIRLAB_SOURCE_PREFIX` in that command's shell to match `cloud`.

## 4. Configure credentials and W&B

For online W&B tracking, set the key in the ignored `.env`:

```dotenv
WANDB_API_KEY=your_private_key
```

The run YAML already contains the project and entity. Set `wandb.mode` to `online`, `offline`, or `disabled`.
W&B logs training losses, learning rates, evaluation metrics, and available RL rewards.
Model uploads and gradient watching are disabled by default.

W&B run names follow `<date>/<model-repository>/answer_only/<stage>/<run-name>`.
Local tracking files stay under the run's `tracking/` directory and are excluded from model publication.
Online resumes reuse the run ID. Offline resumes create separate local sessions.

To upload offline logs later:

```bash
uv run --group rl wandb sync /path/to/run/tracking/wandb/offline-run-*
```

Cloud publication requires write credentials. Choose one authentication method:

| Method | Variables in `.env` |
|---|---|
| OpenStack application credential | `OS_AUTH_URL`, `AIRLAB_APPLICATION_CREDENTIAL_FILE` |
| S3 credentials | `AIRLAB_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

Use the identity URL from your OpenStack configuration.
The application-credential JSON must contain `id`, `secret`, and `project_id`. Keep it outside the repository.
Do not configure both authentication methods.

## 5. Prepare the vLLM server

All recipes use vLLM for evaluation. RL also uses it for rollouts. SFT weight updates run through TRL `SFTTrainer`.

Install the pinned dependencies and apply the video patches on the training and server hosts:

```bash
uv sync --locked --group rl
uv run --group rl python scripts/train/patch_vllm.py --apply
uv run --group rl python scripts/train/patch_vllm.py
```

The patches require the system `patch` command and vLLM 0.30.0.
Apply them again after recreating the environment. Use `--group rl` for training commands to retain this dependency group.

Allocate separate GPUs for training and the server. Their device sets must not overlap.
For four server GPUs, the model profiles specify these starting allocations:

| Model | Tensor parallelism | Data parallelism |
|---|---:|---:|
| Qwen 3.5 9B / Gemma 4 12B | 1 | 4 |
| Qwen 3.8 27B / Gemma 4 31B | 2 | 2 |

Set these shell variables for your allocation:

```bash
export BASE_MODEL_DIR="/path/to/model-weights/weights/Qwen/Qwen3.5-9B/REVISION"
export ROLLOUT_TP=1
export ROLLOUT_DP=4
export ROLLOUT_MAX_SEQS=YOUR_CHECKED_CONCURRENCY
```

Replace `REVISION` with the downloaded revision. Set `ROLLOUT_MAX_SEQS` from the memory and throughput device check.
Server concurrency is separate from the training batch size.

Example server on GPUs 4–7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_SERVER_DEV_MODE=1 uv run --group rl vllm serve "$BASE_MODEL_DIR" \
  --served-model-name triage-policy --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size "$ROLLOUT_TP" \
  --data-parallel-size "$ROLLOUT_DP" --max-num-seqs "$ROLLOUT_MAX_SEQS" \
  --max-model-len 262144 --enable-scale-out --enable-mm-embeds --video-pruning-rate 0 \
  --limit-mm-per-prompt '{"video":1}' \
  --weight-transfer-config '{"backend":"nccl"}'
```

Match `execution.vllm_model_name: triage-policy` and `execution.vllm_server_url` in the run YAML.
For this single-host example, set `execution.training_devices: [0, 1, 2, 3]` and `execution.training_gpus: 4`.
The launcher does not start the server. For separate hosts, use a private reachable address and the matching endpoint.
The weight-transfer interface must not be publicly exposed. It requires unauthenticated control requests on the private network.

The trainer supplies processed video inputs and synchronizes language, vision, and connector weights.
The server does not resample those videos. Start it with full model weights, without a separate LoRA adapter.

## 6. Inspect and launch

Inspect the run YAML:

```bash
uv run --group rl python scripts/train/run.py --config scripts/train/configs/runs/my_run.yaml --inspect
```

This prints the resolved configuration, stage commands, batch sizes, and pending data checks.
It does not read videos, load weights, load credentials, or start W&B. It is not a dataset or GPU validation run.

After preparing data, weights, and the server, launch:

```bash
uv run --group rl python scripts/train/run.py --config scripts/train/configs/runs/my_run.yaml
```

The launcher starts one process per configured training GPU on a single host.
For SFT → RL, RL starts only after SFT succeeds. Each stage has a separate output directory.
Training never uploads model weights automatically.

For resume, set `resume` to `/path/to/run/checkpoints/checkpoint-N` and keep the other run YAML fields unchanged.
The checkpoint identifies the stage. An RL-stage resume does not repeat SFT.
Resume checks the saved configuration, prompts, catalog, and dataset fingerprints.

## 7. Review and publish the result

```text
<output_root>/<date>/<model-repository>/answer_only/<stage>/<run-name>/
├── merged/                    # Complete best model and processor
├── adapter/                   # Best LoRA adapter and processor
├── checkpoints/               # Resume state and evaluation reports
├── tracking/                  # Local W&B files
├── run.json
├── checkpoint_selection.json
├── test_results.json
├── label_catalog.yaml
└── prompts.yaml
```

Stage names are `sft_only`, `rl_only`, and `sft_then_rl`.
Resolved configuration files also live under `<output_root>/.run-configs/`. Saved run metadata excludes credentials.
The split date and run date are independent.

After training, the exporter merges the best adapter into the base weights.
Distributed workers shut down before the CPU export. Reserve host RAM for the complete model and its reload check.
If export fails after saving the adapter, retry:

```bash
uv run --group rl python scripts/train/model_storage.py export --run-dir /path/to/run
```

Inspect publication, then upload:

```bash
uv run --group rl python scripts/train/model_storage.py publish --run-dir /path/to/run --inspect
uv run --group rl python scripts/train/model_storage.py publish --run-dir /path/to/run
```

Publication uses the cloud destination recorded in `run.json`:

```text
<container>/<output-prefix>/<date>/<model-repository>/answer_only/<stage>/<run-name>/
```

The default is `vlm-zoo/finetuned/...`. The run date comes from the run, not the upload date.
Publication retains local files, resumes identical uploads, and refuses to replace different cloud content.
It writes `complete.json` last. Deploy the complete model from `merged/` in a published run with this marker.
Merged shards are limited to 4 GB. The uploader accepts files up to 5 GB.

## Task mixture and grounding

Every enabled valid medical VQA example is retained. `subject_check` is excluded. Medical context questions apply only to native videos.
Auxiliary questions are additional examples, resampled each epoch and shuffled with medical examples across videos and tasks.
There are no fixed per-batch task quotas.

| Auxiliary task | Default sampling |
|---|---|
| Camera viewpoint | One question per eligible native video |
| Counting | One annotated timestamp per eligible native video |
| Subject localization | One human subject point per eligible video |
| Label questions | Two observables and two assessments when both subcategories exist. Otherwise one label |

Label sampling is capped by available labels and prefers a positive/negative pair. It excludes context and subject-check.
For medical-only training, disable `camera_view`, `counting`, `subject_localization`, and `label_questions` in `mixture`.
Tasks with no allowed source in the selected dataset are reported as inactive.

Spatial and temporal switches change the label question's output rather than adding duplicate questions.
SFT and RL share required evidence targets. Negative labels produce empty grounding arrays.
Evidence requirements appear in both positive and negative questions. Human coordinates and evidence times stay out of the question.

Coordinates use normalized `[x, y]` values from 0–1000. Evidence timestamps identify frames. Event spans identify intervals.
Targets use `MM:SS`, retaining fractional seconds only when supported and needed.
Counting/evidence frames and temporal boundary frames stay within the configured sampling budget.
Examples that require more frames than the budget are excluded and reported.

Before enabling stabilized spatial tasks, check the corrected coordinates and set:

```yaml
sft:
  data_checks:
    stabilized_coordinates_verified: true
```

This records a completed check. It does not correct coordinates.
Missing or invalid auxiliary targets are reported while valid medical questions remain available.

<details>
<summary>RL spatial reward configuration</summary>

Configure rewards in the run YAML under `rl.rewards`.
Spatial matching requires the correct label, required anatomical slot, and a timestamp within ±0.5 seconds.
One-to-one matching prevents duplicate credit. Distances use pixels in the human annotation's native or stabilized frame.

| Evidence | Full credit | Zero credit |
|---|---:|---:|
| Default included labels | ≤10 px | ≥20 px |
| Tripod, abnormal head/neck posture, wide-open mouth | ≤20 px | ≥40 px |

Credit decreases linearly between the radii. Tripod placement refers to the casualty's knees.
Subject-type evidence accepts a point inside the designated subject box at the matched frame.
Body-posture and medic-presence label questions omit spatial requests in RL.
Other respiratory labels retain classification and applicable temporal tasks without spatial requests.
These reward rules do not change human annotations. Temporal-span tolerances are configured separately.

</details>

## Training and evaluation reference

The example YAMLs record all values. Current shared defaults include:

| Configuration | Value |
|---|---|
| LoRA | Rank 32, alpha 32 |
| Precision | BF16 base weights, FP32 adapters |
| SFT | Learning rate `1e-4`. Effective batch 16 |
| RL | GSPO. Learning rate `1e-5`. Eight responses per question |
| RL batch | 64 question slots, 512 responses per update |
| Distributed training | FSDP2, activation checkpointing, no CPU offload |
| Video | Up to 10 FPS, Qwen total pixel budget 251,658,240, Gemma 280 soft tokens per frame |

The processor determines spatial resizing. These budgets do not specify a fixed square resolution.
LoRA covers selected language, vision, and connector projections. `run.json` records the exact module list.

All recipes use the same medical evaluator through vLLM, with thinking disabled and category-constrained JSON.
Selection rules are checked after generation. Generation parameters do not control the teacher-forced SFT loss.
Validation uses the generation configuration in the model profile. RL rollout sampling uses the RL configuration.

Best-checkpoint selection uses category-balanced macro F1:

1. Calculate binary F1 per label, including absent and unassessable labels.
2. Omit labels with no positives in either ground truth or predictions.
3. Average within each category, then average categories equally.

Invalid answers count as errors for every expected label. Categories without scorable labels are excluded.
The selected best checkpoint is evaluated on the held-out medical test set. Test scores never select checkpoints.
An empty test set produces an explicit skipped report.

## Checks before a full run

- Dataset: current media, valid answers, aligned coordinates, timestamps, and split integrity.
- Device run: memory, decoding throughput, video forward/backward, weight synchronization, constrained generation, resume, and merged-model reload.
- Each model: training and rollout token positions with the actual checkpoint.

Configuration and CPU checks do not establish GPU memory fit or full-model correctness.
At the full Qwen budget, video activations can exceed an 80 GB GPU. More data-parallel ranks do not solve that limit.
The code does not silently reduce FPS, resolution, or context length.

## Code map and advanced commands

| Path | Purpose |
|---|---|
| `run.py`, `configs/runs/` | Single-YAML launch and examples |
| `prepare.py`, `create_splits.py`, `splits/` | Human targets and split references |
| `shared/` | Video processing, tasks, metrics, configuration, and run metadata |
| `sft/` | TRL SFT trainer and launch |
| `rl/`, `rl/rollout/` | GSPO, rewards, video transport, and weight synchronization |
| `storage/`, `model_storage.py` | Download, merge, and publish |
| `prompts/answer_only/RGB/` | Medical and auxiliary task prompts |

The direct entry points remain available: `train_sft.py`, `train_rl.py`, and `train_sft_then_rl.py`.
Their `--config` argument accepts the older experiment/recipe format, not the complete run YAML.
Use `--help` for their arguments. `train_sft_then_rl.py` runs only the RL stage from an existing SFT run.

Direct SFT `--inspect` reads prepared examples. `--check-batch` also decodes a video through the processor without loading model weights.
Direct RL inspection with dataset paths checks media geometry and reward targets, which can take time.
These checks differ from `run.py --inspect`, which only resolves the run configuration.
