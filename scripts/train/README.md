# Answer-only training

[Repository guide](../../README.md) · [Preprocessing](../preprocess/README.md) · [Verified SFT preflight and limits](PREFLIGHT.md)

Train RGB video-language models on human medical answers with TRL, Transformers, and PEFT LoRA.
Choose SFT, RL only, or SFT followed by RL. Medical answers use category JSON with integer `0`/`1` values.

## Requirements

- Complete the [repository installation](../../README.md#installation).
- Prepare videos and human-answer targets on the data server.
- Provide local storage for data, base weights, checkpoints, and exported models on the training machine.
- Allocate NVIDIA GPUs for training and a separate vLLM server.

Run commands from the repository root. Use one YAML file to configure each run.

## Environment variables

Dataset and storage commands load the repository `.env`. Existing shell values take precedence.
`run.py` loads the file selected by `env_file` in the run YAML.

| Task | Variables | When needed |
|---|---|---|
| GT and split preparation | Dataset path variables below | Only when omitting `--media-root` |
| Base-weight download | `LOCAL_WEIGHTS_ROOT` | Only when omitting `--weights-root`; downloads go under its `weights/` subdirectory |
| W&B tracking | `WANDB_API_KEY` | Online tracking; offline and disabled modes need no key |
| vLLM connection | `VLLM_API_KEY` | Only if the server requires API authentication |
| Cloud upload | [Storage credentials](#upload-the-model) | Uploading model files |

GT and split commands choose the dataset path in this order:

```text
--media-root → LOCAL_DATASET_ROOT → TRAIN_MEDIA_ROOT → AIRLAB_DATASET_ROOT → TRIAGE_DATASET_ROOT
```

Set only the path variable you need, or pass `--media-root` explicitly.
The GT and split commands below need no environment variables or API keys because they include the path.

For training, the run YAML supplies dataset paths, output paths, model selection, and W&B configuration.
Its `cloud` and `wandb` values override matching environment values.
Its `execution.training_devices` sets `CUDA_VISIBLE_DEVICES` for the training processes.
The server command below sets its own GPU allocation separately.

## Prepare the dataset


Complete [preprocessing](../preprocess/README.md) first. Run the following commands on the data server.

GT and split preparation read `dataset_manifest.json` inside `06` and `07`.
Preprocessing publishes these files with the successfully prepared, eligible clips.
Neither command reads `04`, `05`, or the website's live completion list.

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
- Prioritizes retaining positive examples of every available observable label in training.
- Then balances held-out sizes, assessment coverage, year proportions, and observable positives.
- Includes at most one segment per held-out casualty group.
- Excludes other segments of that held-out casualty from all splits.
- Keeps source recordings separate across splits and seeks year and recording diversity.

Labels present in fewer than three recordings cannot have positives in every recording-disjoint split; the report identifies these limits.
Grouping can change the achieved percentages. `summary.json` reports actual sizes, coverage, exclusions, and fingerprints.
Person IDs group casualties within their source recording. They do not identify the same person across different recordings.
Validation and test are deliberately positive-enriched, so their scores do not estimate natural label prevalence.

Each JSONL row references one medical question:

```json
{"variant":"native","clip":"97g7c6735tdk","category":"severe_hemorrhage"}
```

The split files do not duplicate media or answers. Training loads answers from each clip's `GT/medical_vqa.json`.
These commands prepare RGB, answer-only training data.

### Stage data on the training machine

Finish media and GT preparation before transfer. Stop preprocessing and annotation updates while copying.

The transfer script copies:

- Every clip folder listed in the `06` and `07` prepared manifests, including videos, annotations, and existing GT.
- Both prepared manifests.
- The selected split directory, preserving its path under `08-train_val_test`.

This includes prepared clips outside the split and any prepared thermal clips. The split files control which clips training uses.
Stale clip folders that are absent from the manifests are excluded. Files from `04` and `05` are not copied.

Requirements: `rsync` on both machines and SSH access to the destination.
The destination directory must exist. The script does not need API keys or a `.env` file; provide paths through its arguments.

Run these commands on the machine that holds the source dataset.
Set your paths and SSH destination:

```bash
export DATASET_ROOT='/path/to/dataset'
export TRANSFER_DESTINATION='username@transfer-host:/path/to/datasets'
```

Preview the transfer:

```bash
uv run python scripts/train/transfer_dataset.py \
  --media-root "$DATASET_ROOT" \
  --split-dir 08-train_val_test/2026-10-31/answer_only \
  --destination "$TRANSFER_DESTINATION" \
  --inspect
```

The preview checks source files and split checksums, then contacts the destination to compare files.
It lists proposed changes and reports source size and transfer statistics. It copies nothing.

Start the transfer:

```bash
uv run python scripts/train/transfer_dataset.py \
  --media-root "$DATASET_ROOT" \
  --split-dir 08-train_val_test/2026-10-31/answer_only \
  --destination "$TRANSFER_DESTINATION"
```

Rerun the same command after an interruption. Completed files are skipped when their sizes and modification times match.
Partial files are retained for reuse. Changed files are updated; destination files are never deleted.
The script rejects missing inputs, symbolic links, and manifests that differ from those recorded in the split.
It also checks for source metadata changes before and after transfer. A change causes an error; stop updates and rerun.

Set `paths.media_root` to the destination dataset directory in your run YAML.
Set `paths.split_dir` to its `08-train_val_test/<date>/answer_only` directory.
Use the same catalog and prompts as dataset preparation.

Training does not need `04`, `05`, or a live connection to the annotation site.
Keep checkpoints and exports on persistent storage. Training paths remain configurable in the run YAML.

## Configure a run


Use one complete YAML per run. Copy an example within the same directory:

```bash
cp scripts/train/configs/runs/qwen9b_stabilized_medical.yaml scripts/train/configs/runs/my_run.yaml
```

| Example | Model | Videos | Tasks |
|---|---|---|---|
| [qwen9b_stabilized_medical.yaml](configs/runs/qwen9b_stabilized_medical.yaml) | Qwen 3.5 9B | Stabilized | Medical VQA |
| [gemma12b_combined_auxiliary.yaml](configs/runs/gemma12b_combined_auxiliary.yaml) | Gemma 4 12B | Native + stabilized | Medical VQA + auxiliary |

Both examples start with SFT. Larger model profiles are available in [configs/models.yaml](configs/models.yaml).

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

| Recipe | Starting weights | Stages |
|---|---|---|
| `sft` | Verified cloud base weights | SFT |
| `rl_only` | Verified cloud base weights | GSPO with a new LoRA adapter |
| `sft_then_rl` | Verified cloud base weights | SFT, then GSPO from the best SFT adapter |

For `sft_then_rl`, leave `sft_run: null` to run both stages.
To reuse a completed SFT run, set `sft_run` to its directory. The launcher checks compatibility before starting RL.

### W&B tracking


For online W&B tracking, add your key to the local `.env`:

```dotenv
WANDB_API_KEY=your_private_key
```

The run YAML already contains the project and entity. Set `wandb.mode` to `online`, `offline`, or `disabled`.
W&B logs training losses, learning rates, evaluation metrics, and available RL rewards.

W&B run names follow `<date>/<model-repository>/answer_only/<stage>/<run-name>`.
Local tracking files stay under the run's `tracking/` directory and are excluded from model publication.

To upload offline logs later:

```bash
uv run --group rl wandb sync /path/to/run/tracking/wandb/offline-run-*
```

## Download base weights


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
`AIRLAB_SWIFT_URL` overrides the default Swift endpoint.
Without `--weights-root` or `LOCAL_WEIGHTS_ROOT`, `AIRLAB_WEIGHTS_DIR` supplies the base-weight directory directly.
For a different source, set `AIRLAB_SOURCE_CONTAINER` and `AIRLAB_SOURCE_PREFIX` in that command's shell to match `cloud`.

## Start the vLLM server


All recipes use vLLM for evaluation. RL also uses it for rollouts. SFT weight updates run through TRL `SFTTrainer`.

Install the pinned dependencies and apply the video patches on the training and server hosts:

```bash
uv sync --locked --group rl
uv run --group rl python scripts/train/patch_vllm.py --apply
uv run --group rl python scripts/train/patch_vllm.py
```

The patches require the system `patch` command and vLLM 0.30.0.
The Qwen bridge patch prevents duplicate size arguments when vLLM forwards the configured video settings to the Hugging Face processor.
Apply them again after recreating the environment. Use `--group rl` for training commands to retain this dependency group.

If an existing environment has the earlier Gemma patch, stop its serving processes before upgrading.
The current patch checker rejects a partially upgraded source tree.
Retrieve the exact old patch, verify its checksum, and reverse only that patch:

```bash
(
set -e
OLD_GEMMA_PATCH="$(mktemp)"
git show 4ec63f2:scripts/train/rl/rollout/patches/gemma-video-budget.patch > "$OLD_GEMMA_PATCH"
printf '%s  %s\n' 3deaeacd01ca33264d8ab357bf531e4db1e35832b03f2ca53ed7869a10ee4c2d "$OLD_GEMMA_PATCH" | sha256sum --check
VLLM_PACKAGE="$(uv run --group rl python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')"
patch --batch --fuzz=0 --reverse --dry-run -p1 -d "$VLLM_PACKAGE" -i "$OLD_GEMMA_PATCH"
patch --batch --fuzz=0 --reverse -p1 -d "$VLLM_PACKAGE" -i "$OLD_GEMMA_PATCH"
uv run --group rl python scripts/train/patch_vllm.py --apply
uv run --group rl python scripts/train/patch_vllm.py
rm "$OLD_GEMMA_PATCH"
)
```

Do not reverse the video-placeholder patch. If either dry-run or verification fails, stop and inspect the source tree.
Fresh environments use the normal installation commands above.

The locked environment includes `fla-core` for Qwen's gated-delta attention kernels.
Kernel compilation needs a compatible CUDA toolkit and C++ compiler.

Keep `uv` on the server's `PATH`.
In a uv environment without pip, vLLM uses `uv pip list` for its environment report.
Check both `/health` and `/server_info?config_format=json` before loading training models.
A successful health check alone does not verify the environment-report endpoint.

Use writable local storage for compiler caches.
If the account has no home directory, set these paths before starting training or serving:

```bash
export TRIAGE_KERNEL_CACHE=/path/to/writable/node-local/cache
export XDG_CACHE_HOME="$TRIAGE_KERNEL_CACHE/xdg"
export XDG_CONFIG_HOME="$TRIAGE_KERNEL_CACHE/config"
export TRITON_CACHE_DIR="$TRIAGE_KERNEL_CACHE/triton"
export TILELANG_CACHE_DIR="$TRIAGE_KERNEL_CACHE/tilelang"
export FLASHINFER_WORKSPACE_BASE="$TRIAGE_KERNEL_CACHE/flashinfer"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TRITON_CACHE_DIR" \
  "$TILELANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
```

On PSC `cntc-gb001`, the GPU 2/3 communication check stalled with the default NCCL transport.
The same check passed with `NCCL_P2P_DISABLE=1`.
For that node setup, export this setting before starting training and serving.
Do not apply this workaround to other nodes without checking their communication paths.

For long native videos, SFT supports `sft.training.activation_offloading: true`.
This setting does not enable RL activation offloading.
Check memory with representative clips before a full run; the optimizer-step smoke does not establish worst-case memory use.

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
export ROLLOUT_GPU_MEMORY_UTILIZATION=YOUR_CHECKED_FRACTION
```

Replace `REVISION` with the downloaded revision. Set `ROLLOUT_MAX_SEQS` from the memory and throughput device check.
Server concurrency is separate from the training batch size.
For RL, set `ROLLOUT_MAX_SEQS` to at least `rl.policy.num_generations`.
A reduced smoke batch does not verify full-batch capacity.
Check the configured response count and representative video sizes separately.
Reserve GPU memory for weight-transfer buffers and NCCL communicators when choosing the memory fraction.

Install `ffprobe` and add its directory to `PATH` before RL.
Video-geometry checks require this executable, in addition to FFmpeg shared libraries.

Derive the server's profiling flags from the same run and pinned processor used by training.
Set `PROFILE_VIDEO_FRAMES` to the verified maximum sampled frame count for the full dataset.
The frozen PSC answer-only split dated 2026-10-31 has a maximum of 150 frames.
For Gemma, this bound controls startup profiling; it does not truncate input videos or change sampling.
Qwen derives its reservation from the whole-video pixel budget and does not use a frame override.

```bash
RUN_CONFIG=/path/to/run.yaml
PROFILE_VIDEO_FRAMES=150
ROLLOUT_PROCESSOR_KWARGS="$(uv run --group rl python -m scripts.train.rl.rollout.video_profile --run "$RUN_CONFIG" --profile-video-frames "$PROFILE_VIDEO_FRAMES" --field mm_processor_kwargs)"
ROLLOUT_MM_LIMITS="$(uv run --group rl python -m scripts.train.rl.rollout.video_profile --run "$RUN_CONFIG" --profile-video-frames "$PROFILE_VIDEO_FRAMES" --field limit_mm_per_prompt)"
```

The helper loads processor metadata, not model weights. It does not start a server or training.
Qwen uses the full configured video pixel budget and disables per-frame pixel capping, matching the training processor.
For the current budget, its encoder reservation is 122,880 tokens; the frozen metadata maximum is 122,661.
Gemma uses 280 tokens per frame and a profiling image of 960 × 672 pixels.
With 150 frames, its reservation is 43,200 tokens, including delimiters and timestamp allowance.
The updated patch also applies this frame bound to dummy inputs and encoder graph metadata.
These reservations do not prove GPU capacity or throughput. Run the bounded checks on the selected devices.
A healthy server with default processor settings can still reject full videos because its encoder cache is too small.

Example server on GPUs 4–7:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_SERVER_DEV_MODE=1 uv run --group rl vllm serve "$BASE_MODEL_DIR" \
  --served-model-name triage-policy --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size "$ROLLOUT_TP" \
  --data-parallel-size "$ROLLOUT_DP" --max-num-seqs "$ROLLOUT_MAX_SEQS" \
  --gpu-memory-utilization "$ROLLOUT_GPU_MEMORY_UTILIZATION" --logprobs-mode processed_logprobs \
  --max-model-len 262144 --enable-scale-out --enable-mm-embeds --video-pruning-rate 0 \
  --limit-mm-per-prompt "$ROLLOUT_MM_LIMITS" --mm-processor-kwargs "$ROLLOUT_PROCESSOR_KWARGS" \
  --weight-transfer-config '{"backend":"nccl"}'
```

Match `execution.vllm_model_name: triage-policy` and `execution.vllm_server_url` in the run YAML.
For this single-host example, set `execution.training_devices: [0, 1, 2, 3]` and `execution.training_gpus: 4`.
The launcher does not start the server. For separate hosts, use a private reachable address and the matching endpoint.
The weight-transfer interface must not be publicly exposed. It requires unauthenticated control requests on the private network.

The trainer supplies processed video inputs and synchronizes language, vision, and connector weights.
The server does not resample those videos. Start it with full model weights, without a separate LoRA adapter.

## Launch training


Inspect the run YAML:

```bash
uv run --group rl python scripts/train/run.py --config scripts/train/configs/runs/my_run.yaml --inspect
```

Inspection prints the resolved configuration and launch commands without starting training.
It does not validate the dataset or GPU memory fit.

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

Before a full run, check GPU memory, video throughput, weight synchronization, evaluation, and merged-model reload on the target hardware.
The configured video budget can exceed one GPU's memory. Increasing data-parallel workers does not reduce each video's activation memory.

## Retrieve the trained model


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
The exporter rejects missing, extra, or incorrectly shaped adapter tensors before merging.
The export manifest records adapter completeness separately from the offline reload check.
Distributed workers shut down before the CPU export. Reserve host RAM for the complete model and its reload check.
Reload verification compares text logits; it does not verify video inference.

Runtime compatibility helpers are not stored in exported weights.
For Gemma video inference with Transformers 5.21, install the frame-count helper after loading the model.
For CPU Qwen inference with `fla-core` installed, use the attention context around the forward call:

```python
from scripts.train.shared.video_compat import install_video_frame_compatibility
from scripts.train.shared.attention_compat import cpu_export_attention

install_video_frame_compatibility(model)
with cpu_export_attention(model):
    outputs = model(**inputs)
```

These helpers leave other model families unchanged. The training loader and export verification apply them where required.
vLLM uses its own model implementations.

If export fails after saving the adapter, retry:

```bash
uv run --group rl python scripts/train/model_storage.py export --run-dir /path/to/run
```

## Upload the model

Add write credentials to `.env`. Choose one authentication method:

| Method | Variables in `.env` |
|---|---|
| OpenStack application credential | `OS_AUTH_URL`, `AIRLAB_APPLICATION_CREDENTIAL_FILE` |
| S3 credentials | `AIRLAB_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

Use the identity URL from your OpenStack configuration.
The application-credential JSON must contain `id`, `secret`, and `project_id`. Keep it outside the repository.
Configure only the authentication method you use.
For temporary S3 credentials, also set `AWS_SESSION_TOKEN`.
`AWS_DEFAULT_REGION` defaults to `us-east-1`.
Public base-weight downloads need no write credentials.

Inspect the destination, then upload:

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

## Reference

<details>
<summary>Task mixture and grounding</summary>


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

### RL spatial rewards

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

<details>
<summary>Training defaults and evaluation metrics</summary>


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


</details>

<details>
<summary>Code layout and direct entry points</summary>


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

</details>
