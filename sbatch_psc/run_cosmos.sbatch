#!/bin/bash
#SBATCH -N 1 -A cis240120p -p GPU-shared --gpus=h100-80:1 -t 08:00:00 --job-name=cosmos2
#SBATCH --output=cosmos_%j.out
#SBATCH --error=cosmos_%j.err

set -exo pipefail

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR"

# Load existing conda environment
source "$HOME/.bashrc"
if ! command -v conda >/dev/null 2>&1; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "conda not found; please adjust run_cosmos.sbatch to your conda install path"
    exit 1
  fi
fi
conda activate finetune_vlm

export WANDB_API_KEY="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"
export HF_HOME=/ocean/projects/cis240120p/kandiyap/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export WANDB_DIR=/ocean/projects/cis240120p/kandiyap/.cache/wandb
export WANDB_CACHE_DIR=/ocean/projects/cis240120p/kandiyap/.cache/wandb/cache
export WANDB_CONFIG_DIR=/ocean/projects/cis240120p/kandiyap/.cache/wandb/config
mkdir -p "$HF_HUB_CACHE" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# Build cosmos_predictions.json with SFT-CoT first, then append others.
python inference/cosmos_reason_2/inference_cosmos_sft.py
python inference/cosmos_reason_2/inference_cosmos_base.py
python inference/cosmos_reason_2/inference_cosmos_no_cot.py
python inference/cosmos_reason_2/inference_cosmos_rl.py