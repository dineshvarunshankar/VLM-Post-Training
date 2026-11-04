#!/usr/bin/env bash
# One MPI process per node; each starts eight torchrun workers inside the ABCI container.
set -euo pipefail
umask 077
if [[ ${1:-} != inside ]]; then
  repo=$1; config=$2; job_root=$3; master=$4; job=$5; nodes=$6
  rank=${OMPI_COMM_WORLD_RANK:?MPI node rank is missing}
  [[ "$rank" == 0 || "$rank" == 1 ]]
  work=$(cd "$repo/../.." && pwd)
  image=${ABCI_IMAGE:-/groups/gai51740/RBT3/singularity/pytorch-dev-25.08.sif}
  [[ -f "$image" && -x "$work/env/vlm-post-training/bin/python" ]]
  mkdir -p "$job_root/home-$rank"
  exec singularity exec --cleanenv --nv --home "$job_root/home-$rank:$job_root/home-$rank" -B /groups:/groups \
    "$image" bash "$repo/scripts/train/abci_sft_node.sh" inside "$repo" "$config" "$job_root" "$master" "$job" "$rank" "$nodes" \
    > "$job_root/node-$rank.log" 2>&1
fi
shift
repo=$1; config=$2; job_root=$3; master=$4; job=$5; rank=$6; nodes=$7
work=$(cd "$repo/../.." && pwd)
cd "$repo"
export PBS_JOBID="$job" TRIAGE_NODE_RANK="$rank"
export PATH="$work/env/vlm-post-training/bin:$work/tools/ffmpeg/bin:$PATH"
export LD_LIBRARY_PATH="$work/tools/ffmpeg/lib:/opt/hpcx/ucx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$repo" HF_HOME="$work/cache/huggingface" HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR="/tmp/triage-sft-$UID-$job"
export TRIAGE_TRITON_ROOT="$TMPDIR" TRITON_CACHE_DIR="$TMPDIR/triton" TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"
export CUDA_CACHE_PATH="$TMPDIR/cuda" VLLM_CACHE_ROOT="$TMPDIR/vllm"
export TRIAGE_VLLM_GROUP_PORT=51216 VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$VLLM_CACHE_ROOT"
unset WANDB_API_KEY WANDB_BASE_URL
model=$(python -c 'import sys;from scripts.train.shared.run_config import load_run;print(load_run(sys.argv[1])["model"])' "$config")
python - <<'CHECK'
import torch
assert torch.cuda.device_count() == 8, 'Expected eight visible GPUs per node'
assert all('H200' in torch.cuda.get_device_name(i) for i in range(8)), 'Expected H200 nodes'
CHECK
if [[ "$model" == gemma* ]]; then
  python scripts/train/psc/stage_triton.py
  export PYTHONPATH="$TMPDIR/python:$repo" TRIAGE_GEMMA_DYNAMIC_FLEX_CHECK=1
else
  [[ "$model" == qwen* ]]
  python - <<'CHECK'
import torch, fla
from causal_conv1d import causal_conv1d_fn
x=torch.randn(2,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
w=torch.randn(32,4,device='cuda',dtype=torch.bfloat16,requires_grad=True)
y=causal_conv1d_fn(x,w,activation='silu');y.float().square().mean().backward()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
CHECK
fi
if [[ "$rank" == 0 ]]; then
  python -m scripts.train.psc.launch prepare --scheduler pbs --allocated-nodes "$nodes" --expected-model "$model" --config "$config" --job-root "$job_root/run" --master "$master"
fi
# Node zero publishes this only after successful preflight and bundle creation.
deadline=$((SECONDS + 3600))
while [[ ! -f "$job_root/run/plan.json" ]]; do
  (( SECONDS < deadline )) || { echo 'Timed out waiting for launch preflight.' >&2; exit 1; }
  sleep 2
done
weights=$(python -c 'import json,sys;from pathlib import Path;c=json.load(open(sys.argv[1]))["run"];print(Path(c["paths"]["weights_root"])/c["model_settings"]["repository"]/c["model_settings"]["revision"])' "$job_root/run/plan.json")
if [[ "$model" == gemma* ]]; then
  python scripts/train/psc/prefetch_weights.py "$weights"
fi
if [[ "$rank" == 0 ]]; then
  frames=2
  if [[ "$model" == gemma* ]]; then
    CUDA_VISIBLE_DEVICES="" python -m scripts.train.shared.frame_bound --config "$config" --output "$job_root/frame-bound.json"
    frames=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["profile_video_frames"])' "$job_root/frame-bound.json")
  fi
  kwargs=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field mm_processor_kwargs)
  limits=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field limit_mm_per_prompt)
  python -m scripts.train.psc.launch server-arguments --scheduler pbs --config "$config" > "$job_root/server-arguments.txt"
  mapfile -t reasoning_args < "$job_root/server-arguments.txt"
  # Evaluation shares GPU4 (small models) or GPUs4,5 (large models) with training.
  max_length=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["experiment"]["max_length"])' "$job_root/run/plan.json")
  eval_devices=4; eval_tensor_parallel=1
  [[ "$nodes" == 1 ]] || { eval_devices=4,5; eval_tensor_parallel=2; }
  eval_util=$(python -c 'import torch;print(63*2**30/torch.cuda.get_device_properties(4).total_memory)')
  CUDA_VISIBLE_DEVICES="$eval_devices" VLLM_SERVER_DEV_MODE=1 vllm serve "$weights" "${reasoning_args[@]}" \
    --enable-sleep-mode --enforce-eager --served-model-name "$model" --host 0.0.0.0 --port 8000 --dtype bfloat16 \
    --tensor-parallel-size "$eval_tensor_parallel" --max-num-seqs 1 --gpu-memory-utilization "$eval_util" \
    --logprobs-mode processed_logprobs --max-model-len "$max_length" --enable-scale-out --enable-mm-embeds \
    --video-pruning-rate 0 --limit-mm-per-prompt "$limits" --mm-processor-kwargs "$kwargs" \
    --weight-transfer-config '{"backend":"nccl"}' > "$job_root/server.log" 2>&1 &
  server=$!
  trap 'kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true' EXIT
  deadline=$((SECONDS + 3600))
  until curl --connect-timeout 2 --max-time 3 --noproxy '*' -fsS "http://$master:8000/health" >/dev/null; do
    kill -0 "$server"
    (( SECONDS < deadline )) || { echo 'Evaluation server startup timed out.' >&2; exit 1; }
    sleep 5
  done
  touch "$job_root/server-ready"
else
  deadline=$((SECONDS + 3600))
  while [[ ! -f "$job_root/server-ready" ]]; do
    (( SECONDS < deadline )) || { echo 'Timed out waiting for evaluation server.' >&2; exit 1; }
    sleep 5
  done
fi
python -m scripts.train.psc.launch train --scheduler pbs --config "$config" --job-root "$job_root/run" --master "$master"
