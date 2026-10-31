# VLM Post-Training

Prepare triage video datasets, generate annotations, and train video-language models.

## Workflow

| Step | Guide | Current support |
|---|---|---|
| 1. Prepare videos | [Preprocessing](scripts/preprocess/README.md) | Native and stabilized videos, RGB and thermal |
| 2. Prepare targets and splits | [Training data](scripts/train/README.md#2-prepare-the-dataset) | Human-answer RGB medical VQA |
| 3. Train and export | [Training](scripts/train/README.md) | Answer-only SFT, RL-only, and SFT → RL |
| Optional: generate annotations | [CoT generation](scripts/cot/README.md) | Native RGB scene descriptions. Three other prompts are templates |

Answer-only training does not require CoT generation. Full-model GPU validation remains pending.
The previous Unsloth workflow is in [legacy](legacy/README.md).

## Installation

```bash
git clone https://github.com/dineshvarunshankar/VLM-Post-Training.git
cd VLM-Post-Training
```

If UV is not installed, install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, then install the project environment:

```bash
uv python install 3.12
uv sync --locked
```

Video processing also requires FFmpeg and FFprobe. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

Run task commands from the repository root. The training guide includes the additional vLLM installation steps.

## Configuration

For a fresh clone, create the ignored local environment file:

```bash
cp .env.example .env
nano .env
```

Keep an existing `.env` when updating the repository. Keep real keys out of `.env.example` and Git.

| Task | Where to configure it |
|---|---|
| Preprocessing | Dataset path in `.env`, or `--source-root` |
| CoT generation | Dataset path, Gemini key, and generation configuration in `.env` |
| Training | One run YAML for paths, model, recipe, tasks, cloud destination, and W&B |
| Training credentials | `.env` or the shell |

## Dataset layout

```text
<dataset-root>/
├── 00-data_website/medical/completed_clips.csv
├── 04-segmented_snippets/
├── 05-segmented_stabilized_snippets/
├── 06-native_media_dataset/
├── 07-stabilized_media_dataset/
└── 08-train_val_test/<split-date>/answer_only/
```

The writer reads `04` and `05` without modifying them. It writes one folder per clip in `06` and `07`:

```text
<clip_id>/
├── <clip_id>.mp4
├── <clip_id>.json
├── all_annotations/
├── provenance.json
└── GT/
```

`all_annotations/` contains copied annotations. `GT/` contains derived human-answer targets and generated annotations, each with provenance.
Split files reference these clips. They do not duplicate videos.

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/preprocess/` | Video writer and inventory checks |
| `scripts/cot/` | Gemini generation, browser preview, and prompt YAMLs |
| `scripts/train/` | Dataset preparation, run YAMLs, shared training modules, and model storage |
| `legacy/` | Previous Unsloth scripts and their separate environment |
