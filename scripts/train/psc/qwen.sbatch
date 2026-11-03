#!/bin/bash
#SBATCH --job-name=qwen-cot-sft
#SBATCH --account=cis240120p
#SBATCH --partition=GPU
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --gres=gpu:h100-80:8
set -euo pipefail
umask 077
[[ ${SLURM_JOB_NUM_NODES:?Expected a Slurm allocation} == 2 ]]
repo=${TRIAGE_PSC_REPO:?Set the absolute repository path}
config=${TRIAGE_PSC_CONFIG:?Set the absolute finalized run YAML path}
job_base=${TRIAGE_PSC_JOB_ROOT:?Set the absolute new job-output directory}
expected_commit=${TRIAGE_PSC_COMMIT:?Set the reviewed repository commit}
ffmpeg_root=${TRIAGE_PSC_FFMPEG_ROOT:?Set the installed FFmpeg directory}
[[ "$repo" = /* && "$config" = /* && "$job_base" = /* && "$ffmpeg_root" = /* ]]
root="$repo/scripts/train/psc"
cd "$repo"
[[ $(git rev-parse HEAD) == "$expected_commit" ]]
# Refuse stale or locally modified training code in the execution checkout.
git diff --quiet HEAD -- scripts/train
[[ -z $(git ls-files --others --exclude-standard scripts/train) ]]
module load gcc/13.3.1-p20240614 cuda-h100/13.3.1
export PATH="$repo/.venv/bin:$ffmpeg_root/bin:$PATH"
export LD_LIBRARY_PATH="$ffmpeg_root/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$repo" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
export TRIAGE_VLLM_GROUP_PORT=51216
export VLLM_ENGINE_READY_TIMEOUT_S=3600
cache_root="/tmp/triage-$UID-$SLURM_JOB_ID-qwen-eval"
mkdir -p "$cache_root"/{triton,inductor,cuda,vllm}
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
export CUDA_CACHE_PATH="$cache_root/cuda"
export VLLM_CACHE_ROOT="$cache_root/vllm"
unset WANDB_API_KEY WANDB_BASE_URL
# Dependencies are installed and checked before submission.
python -c 'import causal_conv1d, fla'
# Exercise the actual H100 kernel before loading the production models.
python - <<'CHECK'
import torch, fla
from causal_conv1d import causal_conv1d_fn
assert torch.cuda.device_count() == 8, 'Expected an eight-GPU reservation node'
assert 'H100' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
x=torch.randn(2,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
w=torch.randn(32,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=causal_conv1d_fn(x,w,activation='silu')
y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
print('PSC_H100_QWEN_KERNEL_PASSED',flush=True)
CHECK
master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
job_root="$job_base/$SLURM_JOB_ID"
mkdir -p "$job_base"
python "$root/launch.py" prepare --expected-model qwen3.8-27b --config "$config" --job-root "$job_root" --master "$master"
kwargs=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames 2 --field mm_processor_kwargs)
limits=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames 2 --field limit_mm_per_prompt)
python -m scripts.train.psc.launch server-arguments --config "$config" > "$job_root/server-arguments.txt"
mapfile -t reasoning_args < "$job_root/server-arguments.txt"
weights=$(python -c 'import json,sys;from pathlib import Path;c=json.load(open(sys.argv[1]))["run"];print(Path(c["paths"]["weights_root"])/c["model_settings"]["repository"]/c["model_settings"]["revision"])' "$job_root/plan.json")
# First-node GPUs 4,5 serve evaluation only between training phases; all16 GPUs train.
eval_util=$(python -c 'import torch; print(63*2**30/torch.cuda.get_device_properties(4).total_memory)')
CUDA_VISIBLE_DEVICES=4,5 VLLM_SERVER_DEV_MODE=1 vllm serve "$weights" "${reasoning_args[@]}" --enable-sleep-mode --enforce-eager \
 --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 --dtype bfloat16 \
 --tensor-parallel-size 2 --max-num-seqs 1 --gpu-memory-utilization "$eval_util" \
 --logprobs-mode processed_logprobs --max-model-len 262144 --enable-scale-out \
 --enable-mm-embeds --video-pruning-rate 0 --limit-mm-per-prompt "$limits" \
 --mm-processor-kwargs "$kwargs" --weight-transfer-config '{"backend":"nccl"}' \
 > "$job_root/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
health_deadline=$((SECONDS + 3600))
while (( SECONDS < health_deadline )); do
 kill -0 "$server"
 if curl --connect-timeout 2 --max-time 3 --noproxy '*' -fsS "http://$master:8000/health" >/dev/null; then break; fi
 sleep 5
done
curl --connect-timeout 2 --max-time 3 --noproxy '*' -fsS "http://$master:8000/health" >/dev/null
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none --gpu-bind=none \
 python "$root/launch.py" train --config "$config" --job-root "$job_root" --master "$master"
