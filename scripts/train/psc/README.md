# PSC SFT launch scripts

`qwen.sbatch` and `gemma.sbatch` retain the existing PSC layout for Qwen 27B and Gemma 31B.
Each job reserves two nodes with eight H100 GPUs per node.
All sixteen GPUs train, with microbatch one and effective batch sixteen.
Evaluation uses first-node GPUs four and five while training pauses.
The evaluation server sleeps during training. No GPU is reserved exclusively for evaluation.

The launcher rejects a different rank count or disabled GPU sharing.
It does not silently override these configuration fields.
CoT evaluation enables `qwen3` or `gemma4` reasoning parsing, according to the selected model.
The SFT startup check verifies that the server supports the reasoning budget.

## Runtime requirements

Use the validated PSC environment and a clean, committed repository checkout.
The Qwen script checks the installed causal-convolution GPU kernel.
The Gemma script retains dynamic Flex attention, checkpoint RNG preservation, local Triton staging, and sequential weight prefetch.
The Gemma entrypoint requires Torch 2.13.0 and Transformers 5.21.0.
These helpers came from the tested PSC shared-GPU launchers dated 2026-11-02.
Their CoT integration has CPU checks. It still needs a GPU execution check.

The user's project `.env` must contain `WANDB_API_KEY`.
The launcher does not use the shared account's saved W&B credentials.
Select the dataset release and manifest before submission.
The draft configuration paths are not valid production paths.

## Submission inputs

Set these environment variables before calling `sbatch`:

| Variable | Required value |
| --- | --- |
| `TRIAGE_PSC_REPO` | Absolute path to the committed checkout and its `.venv` |
| `TRIAGE_PSC_COMMIT` | Full reviewed commit hash |
| `TRIAGE_PSC_CONFIG` | Absolute path to the finalized model run YAML |
| `TRIAGE_PSC_JOB_ROOT` | Absolute directory for new job bundles and server logs |
| `TRIAGE_PSC_FFMPEG_ROOT` | Installed FFmpeg directory containing `bin` and `lib` |

Keep the finalized YAML outside the clean checkout unless it is committed.
Use distinct run names and output paths for new experiments.
Select the current reservation and walltime explicitly with `sbatch` arguments.
These scripts do not contain an old reservation, fixed node names, or automatic submission.

For example, after final review:

```bash
sbatch --reservation=RESERVATION --time=HH:MM:SS --output=/absolute/logs/qwen-%j.log scripts/train/psc/qwen.sbatch
```

Replace the reservation, walltime, and log path before execution.
Use `gemma.sbatch` for Gemma 31B.
The log directory must exist before submission.

The job bundle records the resolved run, source configuration, and GPU allocation.
Full training requires successful CoT checks for an optimizer update, checkpoint reload, and both output formats.
