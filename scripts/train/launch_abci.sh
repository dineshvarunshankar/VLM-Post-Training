#!/usr/bin/env bash
# Run once inside an ABCI allocation; this script never submits a job.
set -euo pipefail
if [[ $# -ne 2 || ! "$1" =~ ^(check|server|train)$ ]]; then
  echo "Usage: bash scripts/train/launch_abci.sh check|server|train RUN.yaml" >&2
  exit 2
fi
mode=$1
if [[ "$mode" != check ]]; then
  [[ -n "${PBS_JOBID:-}" ]] || { echo 'A PBS allocation is required.' >&2; exit 2; }
  [[ "${RTYPE:-}" == rt_HF ]] || { echo 'This two-GPU launcher requires an exclusive rt_HF allocation.' >&2; exit 2; }
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
tmp=${PBS_LOCALDIR:-$runtime/tmp}
export SINGULARITY_TMPDIR="$tmp"
args=(exec --cleanenv --home "$runtime/home:$runtime/home" -B /groups:/groups)
[[ -z "${PBS_LOCALDIR:-}" ]] || args+=(-B "$PBS_LOCALDIR:$PBS_LOCALDIR")
[[ "$mode" == check ]] || args+=(--nv)
# Pass only operational settings. Credentials are read from the run's project .env.
exec singularity "${args[@]}" "$image" bash --noprofile --norc -s -- \
 "$mode" "$repo" "$work" "$venv" "$config" "$tmp" "${ABCI_PROFILE_VIDEO_FRAMES:-}" "${ABCI_FFMPEG_BIN:-}" <<'CONTAINER'
set -euo pipefail
mode=$1; repo=$2; work=$3; venv=$4; config=$5; tmp=$6; frames=$7; ffmpeg_bin=$8
cd "$repo"
export PATH="$venv/bin:${ffmpeg_bin:+$ffmpeg_bin:}$PATH"
export HOME="$work/runtime/vlm-post-training/home"
export XDG_CONFIG_HOME="$HOME/.config"
export TMPDIR="$tmp" XDG_CACHE_HOME="$work/cache" HF_HOME="$work/cache/huggingface"
export TORCH_HOME="$work/cache/torch" TRITON_CACHE_DIR="$work/cache/triton"
export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH PYTHONHOME WANDB_API_KEY WANDB_BASE_URL
python - "$config" "$mode" <<'PY'
import json, shutil, sys
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values
from scripts.train.shared.run_config import load_run, compile_run
from scripts.train.run import preflight
plan = compile_run(load_run(sys.argv[1]))
c = plan['run']
if c['recipe'] != 'sft' or c['execution']['training_gpus'] != 1:
    raise SystemExit('This launcher supports one-GPU SFT with one separate server GPU.')
if c['execution']['training_devices'] not in (None, [0]):
    raise SystemExit('Use training_devices: null or [0]; the launcher isolates training on GPU 0.')
url = urlsplit(c['execution']['vllm_server_url'])
if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.port != 8000 or url.path not in ('', '/'):
    raise SystemExit('This launcher requires http://127.0.0.1:8000 for its local server.')
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
    if torch.cuda.device_count() < 2:
        raise SystemExit('Two allocated GPUs must be visible before applying the device split.')
print('Launch preflight passed; GPU memory fit is not established by this check.', flush=True)
PY
python scripts/train/patch_vllm.py
if [[ "$mode" == check ]]; then
  python scripts/train/run.py --config "$config" --inspect
elif [[ "$mode" == train ]]; then
  python - <<'PY'
import requests
response = requests.get('http://127.0.0.1:8000/health', timeout=10)
response.raise_for_status()
PY
  export CUDA_VISIBLE_DEVICES=0
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
PY
)
  if [[ "${model[2]}" == gemma* ]]; then
    [[ "$frames" =~ ^[0-9]+$ && "$frames" -ge 2 ]] || { echo 'Set ABCI_PROFILE_VIDEO_FRAMES to the verified dataset bound for Gemma.' >&2; exit 2; }
  else
    frames=2
  fi
  kwargs=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field mm_processor_kwargs)
  limits=$(python -m scripts.train.rl.rollout.video_profile --run "$config" --profile-video-frames "$frames" --field limit_mm_per_prompt)
  export CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1
  exec vllm serve "${model[0]}" --served-model-name "${model[1]}" --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --tensor-parallel-size 1 --data-parallel-size 1 --max-num-seqs 1 \
    --gpu-memory-utilization 0.8 --logprobs-mode processed_logprobs --max-model-len "${model[3]}" \
    --enable-scale-out --enable-mm-embeds --video-pruning-rate 0 \
    --limit-mm-per-prompt "$limits" --mm-processor-kwargs "$kwargs" --weight-transfer-config '{"backend":"nccl"}'
fi
CONTAINER
