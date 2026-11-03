#!/bin/bash
#SBATCH --job-name=gemma-cot-sft
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
export TRIAGE_VLLM_GROUP_PORT=51226
# These directories are node-local; model, dataset, and HF storage stay unchanged.
cache_root="/tmp/triage-gemma-$SLURM_JOB_ID"
export VLLM_CACHE_ROOT="$cache_root/vllm"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TRITON_CACHE_DIR="$cache_root/triton"
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none bash -c 'df -h /tmp; mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"; test -w "$TORCHINDUCTOR_CACHE_DIR"'

# Exact installed Triton package is copied and SHA-256 checked once per node.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none python "$root/stage_triton.py"
export PYTHONPATH="$cache_root/python:$repo"
unset WANDB_API_KEY WANDB_BASE_URL
export TRIAGE_GEMMA_DYNAMIC_FLEX_CHECK=1
python - <<'CHECK'
import torch, transformers
assert torch.__version__.split('+')[0] == '2.13.0' and transformers.__version__ == '5.21.0'
assert torch.cuda.device_count() == 8
assert 'H100' in torch.cuda.get_device_name(0)
assert hasattr(torch._C._dynamo.eval_frame, '_set_lru_cache')
CHECK
master=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
job_root="$job_base/$SLURM_JOB_ID"
mkdir -p "$job_base"
python "$root/launch.py" prepare --expected-model gemma4-31b-it --config "$config" --job-root "$job_root" --master "$master"
cp "$root/launch.py" "$root/gemma_dynamic_flex_entry.py" "$root/checkpoint_rng_patch.py" "$job_root/"
sha256sum "$job_root/launch.py" "$job_root/gemma_dynamic_flex_entry.py" "$job_root/checkpoint_rng_patch.py" > "$job_root/launcher-sha256.txt"
CUDA_VISIBLE_DEVICES="" python "$root/frame_bound.py" --config "$config" --output "$job_root/frame-bound.json" > "$job_root/frame-bound.log"
frames=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["profile_video_frames"])' "$job_root/frame-bound.json")
kwargs=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field mm_processor_kwargs)
limits=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field limit_mm_per_prompt)
python -m scripts.train.psc.launch server-arguments --config "$config" > "$job_root/server-arguments.txt"
mapfile -t reasoning_args < "$job_root/server-arguments.txt"
weights=$(python -c 'import json,sys;from pathlib import Path;c=json.load(open(sys.argv[1]))["run"];print(Path(c["paths"]["weights_root"])/c["model_settings"]["repository"]/c["model_settings"]["revision"])' "$job_root/plan.json")
# Sequential reads warm the immutable model shards before mmap/FSDP materialization.
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none python "$root/prefetch_weights.py" "$weights" > "$job_root/prefetch.log"
# First-node GPUs 4,5 serve evaluation only between training phases; all16 GPUs train.
eval_util=$(python -c 'import torch; print(63*2**30/torch.cuda.get_device_properties(4).total_memory)')
CUDA_VISIBLE_DEVICES=4,5 VLLM_SERVER_DEV_MODE=1 vllm serve "$weights" "${reasoning_args[@]}" --enable-sleep-mode \
 --enforce-eager --served-model-name gemma4-31b-it --host 0.0.0.0 --port 8000 --dtype bfloat16 \
 --tensor-parallel-size 2 --max-num-seqs 1 --gpu-memory-utilization "$eval_util" \
 --logprobs-mode processed_logprobs --max-model-len 262144 --enable-scale-out \
 --enable-mm-embeds --video-pruning-rate 0 --limit-mm-per-prompt "$limits" \
 --mm-processor-kwargs "$kwargs" --weight-transfer-config '{"backend":"nccl"}' \
 > "$job_root/server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
 kill -0 "$server"
 if curl --connect-timeout 2 --max-time 3 --noproxy '*' -fsS "http://$master:8000/health" >/dev/null; then break; fi
 sleep 5
done
curl --connect-timeout 2 --max-time 3 --noproxy '*' -fsS "http://$master:8000/health" >/dev/null
srun --kill-on-bad-exit=1 --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none --gpu-bind=none \
 python "$job_root/launch.py" train --config "$config" --job-root "$job_root" --master "$master"
