These four CoT drafts preserve the existing medical split policy and add one scene caption per selected video.
Medical targets contain reasoning and final human-label JSON.
Scene captions and native camera targets use direct answers.
Answer-only configurations keep their existing behavior.

Qwen 9B and 27B use an eight-epoch limit with early-stopping patience two.
Each epoch ends with validation. Two consecutive evaluations without higher medical macro-F1 stop training.
Training retains the checkpoint with the highest medical macro-F1.
The callback counter is restored when training resumes from a checkpoint.
Gemma 12B and 31B use learning rate `2e-5` and retain five epochs.
These settings require the model-specific GPU checks before a full CoT launch.

SFT validation and final-test reports now separate medical value tokens from other JSON tokens.
The metrics are `medical_value_loss`, `medical_value_accuracy`, `medical_format_loss`, and `medical_format_accuracy`, with `eval_` or `test_` prefixes.
Each group also reports its token count. An empty group has no loss or accuracy value.
These are teacher-forced token measurements, not generated label accuracy or medical F1.
Reasoning, prompts, native control tokens, captions, and camera answers are excluded from these groups.
A token that contains a medical value and punctuation belongs to the value group.
The metrics use the existing medical teacher-loss pass and do not change the training objective.
CoT teacher metrics cover only accepted teacher targets; generated medical F1 still covers all selected human rows.

No dataset release is selected in these drafts.
The medical and caption generation budgets are 4096 and 1024 tokens.
Medical generation uses the native vLLM `thinking_token_budget: 3064`.
The configuration reserves 1024 tokens for the final answer and eight tokens as a conservative transition allowance.
Native markers and prompt continuation tokens also consume this budget.
The allowance is not a measured uniform tokenizer cost.
The training context remains 262144 tokens, and full teacher targets remain unchanged.
These values require corpus checks before a GPU pilot.
The approved large-model recipes use 16 training GPUs, batch size 16, and microbatch size 1.
Gemma 31B retains `flex_attention` and the shared evaluation-server layout.

Run CPU inspection with an explicit dataset root, frozen split, and pinned local processor files:

```bash
python -m scripts.train.shared.cot_inspect \
  --config scripts/train/configs/cot/qwen3.5-9b.yaml \
  --media-root /explicit/dataset-release \
  --split-dir /explicit/frozen-split \
  --processor-metadata /explicit/pinned-processor-files \
  --output-dir /explicit/new-inspection-directory
```

This command uses the SFT launcher data route and the actual model processor.
It loads no model weights and starts no GPU job.
The other model YAML files use the same command.
Each command requires the corresponding pinned processor files.

The inspection directory contains these immutable files:

- `inspection.html` shows actual student inputs separately from private supervised targets.
- `inspection.json` records coverage, target lengths, token totals, and context diagnostics.
- `source-manifest.json` pins target files, source files, prompt files, and split membership.
- `configuration.json` records the resolved configuration and model revisions.

The command exits with status 2 when training coverage is incomplete.
The report identifies every missing or invalid target.
Existing targets with another source version require a compatibility review.
The loader does not replace missing reasoning with answer-only medical targets.
Held-out medical F1 uses all selected human labels, including rows without accepted teacher reasoning.
Teacher loss uses only accepted teacher rows and reports its coverage.
Caption loss and camera accuracy remain separate from medical F1.
Generation sampling never uses frame indices from held-out human evidence.

The source adapter supports the exact generator profiles in `generator-profiles-2026-11-02.json`.
The versioned files under `teacher-prompts/2026-11-02/` reproduce teacher inputs for these profiles.
They are private generation references, separate from the student instructions in `prompts.yaml`.
The adapter reconstructs teacher inputs and checks their hash, human labels, evidence, source hashes, and recorded duration.
Unknown implementation maps remain unsupported.
An empty `accepted_implementation_fingerprints` list permits the audited maps.
A nonempty list restricts acceptance to the selected audited maps.
It does not permit arbitrary implementations.

Inspection measures all accepted training targets against the configured generation budgets.
It checks a real video context for one example per task.
The collator checks every later batch and rejects context overflow or an excessive evidence-frame budget.
The inspection report does not establish visual correctness or context coverage for the full corpus.
The local HTML artifact supports review without changes to the active port 8767 service.

Before a training review, set `cot.release_manifest` to the reviewed manifest path.
Set `cot.release_manifest_sha256` to the file hash that inspection prints.
Set the regenerated media and split paths. Weight and output roots are preset for ABCI (9B/12B) and PSC (27B/31B).
The four new run identities end in `-medical-scene-camera-cot-1`; they do not reuse answer-only output directories.
Training checks the current sources against the complete manifest.
Manifest identity uses relative `dataset/`, `repository/`, and `splits/` namespaces.
Physical roots remain separate diagnostics, so identical files retain their release identity across clusters.
Distributed training performs this source check on rank zero and broadcasts the result.
Training metadata, output paths, and W&B names record `cot`.
Training logs record supervised token totals by task since the current process started.

GPU execution requires separate authorization and the established model-specific runtime.
Gemma 31B on PSC requires the tested `gemma_dynamic_flex_entry.py` and `checkpoint_rng_patch.py` helpers.
Their versioned source location is `scripts/train/psc/`.
That runtime uses pinned Torch 2.13 and Transformers 5.21, dynamic Flex attention, and safe checkpoint replay.
The YAML attention value alone does not reproduce that runtime.
The GPU pilot must preserve its tested entrypoint and shared-server sleep layout.
The ABCI server path adds `--reasoning-parser qwen3` or `--reasoning-parser gemma4` for CoT only.
The PSC batch scripts add the parser flag from the run configuration.
External PSC server commands require the matching parser flag explicitly.
CoT startup rejects a server without initialized reasoning markers through `/server_info?config_format=json`.
Medical requests send the budget inside `sampling_params`.
Caption, camera, answer-only, and RL requests omit that field.
Run metadata and `cot_generation_policy.json` record the resolved task policies.
A GPU update, checkpoint reload, and both output formats still require separate validation for each model.
PSC launch instructions and runtime requirements are in `scripts/train/psc/README.md`.
