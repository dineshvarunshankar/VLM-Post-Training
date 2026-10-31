#!/bin/bash
#SBATCH -N 1 -A cis240120p -p GPU-shared --gpus=h100-80:2 -t 08:00:00 --job-name=gemma4
#SBATCH --output=gemma4_%j.out
#SBATCH --error=gemma4_%j.err

set -exo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

# Load uv environment
source "$HOME/.bashrc"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; please install it or add it to your PATH"
  exit 1
fi

export HF_HOME=/ocean/projects/cis240120p/kandiyap/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export WANDB_DIR=/ocean/projects/cis240120p/kandiyap/.cache/wandb
export WANDB_CACHE_DIR=/ocean/projects/cis240120p/kandiyap/.cache/wandb/cache
export WANDB_CONFIG_DIR=/ocean/projects/cis240120p/kandiyap/.cache/wandb/config
mkdir -p "$HF_HUB_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

uv run python inference/gemma4/inference_gemma4_sft.py
uv run python inference/gemma4/inference_gemma4_base.py
