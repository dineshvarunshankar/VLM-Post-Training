#!/bin/bash
#SBATCH -N 1 -p GPU-shared --gpus=h100-80:1 -t 01:00:00 --job-name=cosmos_r2_lora
#SBATCH --output=slurm_logs/cosmos_%j.out
#SBATCH --error=slurm_logs/cosmos_%j.err

mkdir -p slurm_logs
cd $SLURM_SUBMIT_DIR

# Install uv if not available
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

# export WANDB_API_KEY="your_key_here"

uv sync
uv run finetune_cosmosreason2_GRPO.py
