#!/usr/bin/env bash
# Run once inside an ABCI allocation; this script never submits a job.
set -euo pipefail
if [[ $# -ne 2 || ! "$1" =~ ^(check|server|train|frame-bound)$ ]]; then
  echo "Usage: bash scripts/train/launch_abci.sh check|server|train|frame-bound RUN.yaml" >&2
  exit 2
fi
mode=$1
train_gpu=${ABCI_TRAIN_GPU:-0}
eval_gpu=${ABCI_EVAL_GPU:-1}
[[ "$train_gpu" =~ ^[0-7](,[0-7])*$ && "$eval_gpu" =~ ^[0-7]$ ]] || { echo 'GPU indices must be integers from 0 to 7.' >&2; exit 2; }
IFS=, read -r -a training_devices <<< "$train_gpu"
declare -A used_devices=()
for device in "${training_devices[@]}"; do
  [[ -z "${used_devices[$device]:-}" ]] || { echo 'Training GPU list contains duplicate indices.' >&2; exit 2; }
  used_devices[$device]=1
done
[[ -z "${used_devices[$eval_gpu]:-}" ]] || { echo 'Training and evaluation GPUs must be distinct.' >&2; exit 2; }
if [[ "$mode" != check && "$mode" != frame-bound ]]; then
  [[ -n "${PBS_JOBID:-}" ]] || { echo 'A PBS allocation is required.' >&2; exit 2; }
  [[ "${RTYPE:-}" == rt_HF ]] || { echo 'This launcher requires an exclusive rt_HF allocation.' >&2; exit 2; }
  [[ -d "${PBS_LOCALDIR:-}" && -w "$PBS_LOCALDIR" ]] || { echo 'PBS_LOCALDIR must be writable.' >&2; exit 2; }
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
work=$(cd "$repo/../.." && pwd)
config=$(realpath "$2")
image=${ABCI_IMAGE:-/groups/gai51740/RBT3/singularity/pytorch-dev-25.08.sif}
venv="$work/env/vlm-post-training"
runtime="$work/runtime/vlm-post-training"
[[ -f "$config" && -f "$image" && -x "$venv/bin/python" ]] || { echo 'Run config, container, or project environment is missing.' >&2; exit 2; }
umask 077
mkdir -p "$runtime/home" "$runtime/tmp" "$work/cache/huggingface" "$work/cache/torch" "$work/cache/triton"
# Keep vLLM Unix socket paths below the platform limit for long run names.
run_tmp_id=$(printf %s "$config" | sha256sum | cut -c1-12)
tmp="${PBS_LOCALDIR:-$runtime/tmp}/$run_tmp_id/$mode"
mkdir -p "$tmp"
export SINGULARITY_TMPDIR="$tmp"
args=(exec --cleanenv --home "$runtime/home:$runtime/home" -B /groups:/groups)
[[ -z "${PBS_LOCALDIR:-}" ]] || args+=(-B "$PBS_LOCALDIR:$PBS_LOCALDIR")
[[ "$mode" == check || "$mode" == frame-bound ]] || args+=(--nv)
# Pass only operational settings. Credentials are read from the run's project .env.
exec singularity "${args[@]}" "$image" bash --noprofile --norc -s -- \
 "$mode" "$repo" "$work" "$venv" "$config" "$tmp" "${ABCI_PROFILE_VIDEO_FRAMES:-}" "${ABCI_FFMPEG_BIN:-}" "$train_gpu" "$eval_gpu" "${ABCI_SERVER_PORT:-}" "${TRIAGE_VLLM_GROUP_PORT:-51216}" <<'CONTAINER'
set -euo pipefail
mode=$1; repo=$2; work=$3; venv=$4; config=$5; tmp=$6; frames=$7; ffmpeg_bin=$8; train_gpu=$9; eval_gpu=${10}; expected_port=${11}
export TRIAGE_VLLM_GROUP_PORT=${12}
cd "$repo"
export PATH="$venv/bin:${ffmpeg_bin:+$ffmpeg_bin:}$PATH"
export HOME="$work/runtime/vlm-post-training/home"
export XDG_CONFIG_HOME="$HOME/.config"
export TMPDIR="$tmp" XDG_CACHE_HOME="$work/cache" HF_HOME="$work/cache/huggingface"
export TORCH_HOME="$work/cache/torch" TRITON_CACHE_DIR="$work/cache/triton/$(basename "$config" .yaml)/$mode"
export LD_LIBRARY_PATH="${ffmpeg_bin:+$ffmpeg_bin/../lib:}/opt/hpcx/ucx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME WANDB_API_KEY WANDB_BASE_URL
if [[ "$mode" == frame-bound ]]; then
  # Read selected inventory metadata before server launch; no decoding, model loading, or patching.
  CUDA_VISIBLE_DEVICES="" exec python -m scripts.train.shared.frame_bound --config "$config" --output "$tmp/frame-bound.json"
fi
python - "$config" "$mode" "$train_gpu" "$eval_gpu" "$expected_port" <<'PY'
import json, shutil, sys
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values
from scripts.train.shared.run_config import load_run, compile_run
from scripts.train.run import preflight
plan = compile_run(load_run(sys.argv[1]))
c = plan['run']
training_devices = [int(index) for index in sys.argv[3].split(',')]
if c['recipe'] != 'sft' or c['execution']['training_gpus'] != len(training_devices):
    raise SystemExit('SFT training_gpus must match the assigned training GPU count.')
if c['execution']['training_devices'] not in (None, training_devices):
    raise SystemExit('Use training_devices: null or the physical training GPU from the allocation plan.')
url = urlsplit(c['execution']['vllm_server_url'])
if url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port or not 1024 <= url.port <= 65535 or url.path not in ('', '/') or url.query or url.fragment or url.username:
    raise SystemExit('This launcher requires an unprivileged HTTP port on 127.0.0.1.')
if sys.argv[5] and url.port != int(sys.argv[5]):
    raise SystemExit('Run YAML HTTP port differs from the allocation plan.')
from scripts.train.shared.vllm import weight_transfer_port
if weight_transfer_port() == url.port:
    raise SystemExit('HTTP and weight-transfer ports must be distinct.')
preflight(plan)
env_path = Path(c['env_file']) if c.get('env_file') else None
if not env_path or env_path.resolve() != Path('.env').resolve():
    raise SystemExit('Use this repository’s project .env for credentials.')
if not dotenv_values(env_path).get('WANDB_API_KEY'):
    raise SystemExit('Add your WANDB_API_KEY to the project .env. Shared-account credentials are not used.')
model = Path(c['paths']['weights_root']) / c['model_settings']['repository'] / c['model_settings']['revision']
if not (model/'config.json').is_file() or not list(model.glob('*.safetensors')):
    raise SystemExit('Pinned model weights are missing.')
for index in model.glob('*.safetensors.index.json'):
    for name in set(json.loads(index.read_text())['weight_map'].values()):
        if not (model/name).is_file() or (model/name).stat().st_size == 0:
            raise SystemExit('An indexed weight shard is missing or empty.')
aux = any(c['mixture'][task]['enabled'] for task in ('camera_view','counting','subject_localization','label_questions'))
if aux and not shutil.which('ffprobe'):
    raise SystemExit('Auxiliary evaluation needs ffprobe. Set ABCI_FFMPEG_BIN to a project-owned container-compatible installation.')
if sys.argv[2] != 'check':
    import torch
    if torch.cuda.device_count() <= max(*training_devices, int(sys.argv[4])):
        raise SystemExit('Assigned GPU indices are not visible in this allocation.')
print('Launch preflight passed; GPU memory fit is not established by this check.', flush=True)
PY
python scripts/train/patch_vllm.py
server_port=$(python - "$config" <<'PY'
import sys
from urllib.parse import urlsplit
from scripts.train.shared.run_config import load_run
print(urlsplit(load_run(sys.argv[1])['execution']['vllm_server_url']).port)
PY
)
if [[ "$mode" == check ]]; then
  python scripts/train/run.py --config "$config" --inspect
elif [[ "$mode" == train ]]; then
  python - "$server_port" <<'PY'
import requests, sys
response = requests.get(f'http://127.0.0.1:{sys.argv[1]}/health', timeout=10)
response.raise_for_status()
PY
  # Variable video shapes otherwise leave large unusable gaps in reserved memory.
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export CUDA_VISIBLE_DEVICES="$train_gpu"
  exec python scripts/train/run.py --config "$config"
else
  # Gemma needs a measured maximum frame count; Qwen does not use the frame override.
  mapfile -t model < <(python - "$config" <<'PY'
import sys
from pathlib import Path
from scripts.train.shared.run_config import load_run, compile_run
c = compile_run(load_run(sys.argv[1]))['run']
print(Path(c['paths']['weights_root']) / c['model_settings']['repository'] / c['model_settings']['revision'])
print(c['execution']['vllm_model_name'])
print(c['model'])
print(c['shared']['max_length'])
from scripts.train.shared.cot_budget import server_arguments
for argument in server_arguments(c):
    print(argument)
PY
)
  server_args=("${model[@]:4}")
  if [[ "${model[2]}" == gemma* ]]; then
    # Gemma graph replay fails after live weight transfer on the current vLLM build.
    server_args+=(--enforce-eager)
    [[ "$frames" =~ ^[0-9]+$ && "$frames" -ge 2 ]] || { echo 'Set ABCI_PROFILE_VIDEO_FRAMES to the verified dataset bound for Gemma.' >&2; exit 2; }
  else
    frames=2
  fi
  kwargs=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field mm_processor_kwargs)
  limits=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field limit_mm_per_prompt)
  export CUDA_VISIBLE_DEVICES="$eval_gpu" VLLM_SERVER_DEV_MODE=1
  exec vllm serve "${model[0]}" "${server_args[@]}" --served-model-name "${model[1]}" --host 127.0.0.1 --port "$server_port" \
    --dtype bfloat16 --tensor-parallel-size 1 --data-parallel-size 1 --max-num-seqs 1 \
    --gpu-memory-utilization 0.8 --logprobs-mode processed_logprobs --max-model-len "${model[3]}" \
    --enable-scale-out --enable-mm-embeds --video-pruning-rate 0 \
    --limit-mm-per-prompt "$limits" --mm-processor-kwargs "$kwargs" --weight-transfer-config '{"backend":"nccl"}'
fi
CONTAINER
