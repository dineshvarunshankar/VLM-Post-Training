# SFT preflight results — 2026-10-31

The bounded SFT pipeline tests passed for all four models. This verifies execution, not full-training capacity or model quality.
The remaining three models passed at code revision `c8d6387`. Qwen9B passed earlier at `6415f29`; later compatibility checks cover its shared code paths.

| Model | Complete SFT test data | Training GPUs | Evaluation GPUs | Evaluation memory fraction |
|---|---|---:|---:|---:|
| Qwen3.5-9B | Selected stabilized clips | 1 H100 80 GB | 1 H100 80 GB | 0.80 |
| Gemma4-12B-it | Selected native and stabilized clips | 2 RTX PRO 6000 96 GB | 1 RTX PRO 6000 96 GB | 0.80 |
| Qwen3.8-27B | Selected native and stabilized clips | 2 RTX PRO 6000 96 GB | 2 RTX PRO 6000 96 GB | 0.70 |
| Gemma4-31B-it | Selected native and stabilized clips | 2 RTX PRO 6000 96 GB | 2 RTX PRO 6000 96 GB | 0.85 |

The Qwen9B complete test passed earlier on H100s. Its later native-video optimizer test also passed, but its combined-data end-to-end retry was not completed.
The other three complete tests used two optimizer steps, effective batch two, and SFT activation offloading.
Their video settings remained 10 FPS, Qwen total pixel budget 251658240, and Gemma 280 soft tokens per frame.
The context limit remained 262144. Server profiling used the flags from `rl.rollout.video_profile`.
Each of these three servers also passed two simultaneous requests with eight responses each and a 1024-token completion limit.

## Verified results

All complete tests included training, validation checkpoint selection, held-out evaluation, adapter saving, merged export, and offline CPU reload.
The remaining three runs also passed strict file-hash, finite-tensor, selected-checkpoint, and parameter-update checks:

| Model | Adapter tensors | Tensors changed from checkpoint 1 to 2 | Exported files rehashed |
|---|---:|---:|---:|
| Gemma4-12B-it | 660 | 330 | 14 |
| Qwen3.8-27B | 1212 | 606 | 22 |
| Gemma4-31B-it | 1202 | 601 | 24 |

These runs selected checkpoint 1 because validation scores tied. Parameter-update proof compares the retained checkpoints 1 and 2.
A tiny validation score does not establish learning quality. Offline full-model export verification compares text logits, not video logits.

## Committed fixes

The tested code includes complete checkpoint-wrapped adapter export, CPU Qwen attention compatibility, and the pinned FLA dependency.
It also preserves Gemma video frame counts through regular, fused, and checkpointed forwards.
The vLLM patches cover video placeholders, Gemma video profiling, and duplicate Qwen processor arguments.
Gemma Unified rollout transfer restores checkpoint weight names. Audio-free Gemma FSDP wrapping excludes the absent optional audio block.
Tracking directory creation occurs after distributed Trainer initialization.

## Fresh environment

Follow the [training setup and server instructions](README.md#start-the-vllm-server). Cloning alone does not prepare the runtime.

1. Obtain the tested code and pinned model revisions. Verify the prepared dataset and split files.
2. Run `uv sync --locked --group rl` and apply all three vLLM patches with `scripts/train/patch_vllm.py --apply`.
3. Run the patch checker again without `--apply`.
4. Configure working FFmpeg/ffprobe, a compatible CUDA/C++ toolchain, and writable local compiler caches.
5. Derive server profiling flags from the same run YAML and processor. Verify both server health and server-info endpoints.
6. Set the actual training and server device lists, data paths, output paths, and tracking credentials for that host.

The private-node tests required `NCCL_P2P_DISABLE=1`; this is a measured workaround for that node, not a universal GPU setting.
Plain Hugging Face Gemma video inference must install the documented frame compatibility helper after loading merged weights.
CPU Qwen inference with FLA needs the documented CPU attention context. These runtime helpers are not serialized in model weights.

## Remaining limits

Full-training batch accumulation, full-dataset worst-video GPU memory, and sustained throughput are not verified by these tests.
Qwen27 and Gemma31 recovered from allocator retries during the small tests; their full-training memory headroom is not established.
CPU video/context checks do not prove GPU fit. Resource fractions above describe the tested host and workloads, not portable defaults.

Remaining-model RL tests were held. An eight-response serving test does not verify an RL optimizer update or the full 512-response batch.
No full training run was launched during this preflight.
