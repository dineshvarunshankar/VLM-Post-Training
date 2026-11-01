# Generate video annotations

[Repository guide](../../README.md) · [Preprocessing](../preprocess/README.md)

Generate RGB scene descriptions and medical spatiotemporal reasoning using Gemini through LiteLLM.
Save the result alongside the clip for review and dataset preparation.

## Requirements

- Complete [preprocessing](../preprocess/README.md) to publish the `06` and `07` prepared manifests.
- Select a published RGB clip with a camera viewpoint. Scene tasks also require a scene transcription.
- Scene tasks use answered medical labels, definitions, and available positive-label spans as consistency context.
- Medical tasks use the repository RGB catalog, assessment prompts, and copied human labels and evidence.
- Install `ffprobe` and `ffmpeg` for medical evidence frames.
- Provide a CMU AI Gateway key with access to the selected model.

Run commands from the repository root.

## Environment variables

Both commands load the repository `.env` automatically. Existing shell values take precedence.
Add these values for generation:

```dotenv
TRIAGE_DATASET_ROOT="/path/to/dataset-root"
GEMINI_API_KEY=your_gateway_key
GEMINI_API_BASE=https://ai-gateway.andrew.cmu.edu/v1
GEMINI_MODEL=gemini/gemini-4-argon
GEMINI_REASONING_EFFORT=high
GEMINI_MAX_OUTPUT_TOKENS=32768
```

Replace the key and model name with values for your account. The model route must start with `gemini/`.
When `GEMINI_API_BASE` is set, LiteLLM sends inline video through that gateway using the supplied key.
No direct Google key or Files API upload is used. For direct Google access, omit `GEMINI_API_BASE` and provide a direct Gemini key.
Preview and inspection need only the dataset path; they do not require an API key.
`--dataset-root` overrides environment paths. Otherwise, `AIRLAB_DATASET_ROOT` takes precedence over `TRIAGE_DATASET_ROOT`.
Leave `AIRLAB_DATASET_ROOT` unset when using `TRIAGE_DATASET_ROOT`.
See [generation options](#generation-options) for the remaining variables and defaults.

## Review an input

Print one clip's complete prompt, response schema, and generation configuration:

```bash
uv run python scripts/cot/generate.py --limit 1 --inspect
```

For a browser preview:

```bash
uv run python scripts/cot/preview.py --port 8765
```

When working on a remote server, forward the port from your computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Open **http://localhost:8765** and select a clip. VS Code Remote SSH can also forward port `8765` through its Ports panel.
The page supports all four tasks. Select a task and clip; medical tasks also have a category selector.
The preview uses the generator's input assembly and response schema.
Video and evidence-image URIs are placeholders. Preview performs no upload or API call.
Medical previews show the associated evidence-frame timestamps; generation attaches those frames as PNG images.
The preview reports schema and medical grounding checks separately from annotation-source phrase flags.
These checks do not verify visual claims or consistency between tasks. The trial UI does not save approvals or edits.

The server binds to `127.0.0.1`. If port `8765` is already occupied, choose another port such as `8767` in both the server command and the forwarding configuration.

### Inputs sent to Gemini

| Input | Source |
|---|---|
| Video with the designated subject box | `<clip_id>.mp4` |
| Modality, duration, time of day, human transcription | `all_annotations/meta.json` |
| Camera viewpoint | `all_annotations/camera_motion.json` |
| Recorded clip-wide counts and subject type | `all_annotations/counting.json` |

Counts are identified as clip-wide totals, not simultaneous counts at a particular timestamp.
The runner checks prepared-file hashes against `provenance.json` before generation.
Clip selection uses the prepared manifests, not a scan of output folders or the live website.
The runner requires RGB, copied completion status, and matching writer provenance. Stale output folders are excluded.
Stabilized tasks do not receive counts. Their coordinates refer to the stabilized frame.

Medical requests include one category's question, definitions, rules, human answer, transcript, and evidence.
The runner converts pixel coordinates to normalized integer `[x,y]` values in `[0,1000]`.
Point timestamps come from the decoded frames identified by human frame numbers. Original annotations remain unchanged.
Human spans retain their annotated boundaries, formatted as `MM:SS` with milliseconds where needed.
Prose can use precise source-frame timestamps. Uncertain event times remain approximate.
Tracked points are supplied as supporting input. Output point references must match supplied human evidence.
Medical evidence references are restricted to positive labels: points require `evidence_min > 0`, and spans require `temporal: true`.
The response schema requires the exact category label keys and binary values.
Evidence images show the prepared video frames at human evidence locations; they do not replace temporal video context.

## Generate one clip

Choose `--task` from the table below. Replace `CLIP_ID` with a published ID in the corresponding dataset.
For example, inspect one stabilized medical request:

```bash
uv run python scripts/cot/generate.py \
  --task stabilized_medical_spatiotemporal_vqa \
  --clip CLIP_ID --category severe_hemorrhage --inspect
```

Generate one native scene description:

```bash
uv run python scripts/cot/generate.py --clip CLIP_ID --generate
```

This sends the prepared MP4 in a paid API request and saves the result.
Review one result before generating a batch.

## Generate a batch

Generate all four tasks:

```bash
uv run python scripts/cot/generate.py --tasks all --generate --workers 4
```

Generate only the two medical tasks:

```bash
uv run python scripts/cot/generate.py \
  --tasks native_medical_spatiotemporal_vqa stabilized_medical_spatiotemporal_vqa \
  --generate --workers 4
```

Use `--tasks` with any task list, or use `--task` for one task. Do not combine the two options.
Tasks run sequentially in the listed order. Each task processes clips concurrently, using the configured worker count.
Duplicate task names run once. A failed task completes its clip batch, then stops the run before the next task.

Preview all tasks without API calls or output writes:

```bash
uv run python scripts/cot/generate.py --tasks all --limit 1 --inspect
```

Generate a limited batch for the default task:

```bash
uv run python scripts/cot/generate.py --limit 10 --generate --workers 4
```

Omit `--limit` to process all published RGB clips for each selected task. The limit applies per variant, not across the run.
The same variant's tasks use the same selected clips. Native and stabilized limits do not guarantee paired clips.
Use `--clip` only with tasks from one variant; native and stabilized clip IDs differ.
Use `--category` only when all selected tasks are medical tasks.
The default task is `native_general_scene_description`. For medical tasks, omit `--category` to process all valid categories.
Subject check is excluded. Context is included only for native medical VQA.
The default is four workers, capped by available CPUs and the selected clip count. Use `--workers 1` for sequential requests.
Matching results are skipped. Failed clips are reported without stopping the rest of the batch.

## Output

```text
06-native_media_dataset/<clip_id>/GT/general_scene_description.json
07-stabilized_media_dataset/<clip_id>/GT/general_scene_description.json
06-native_media_dataset/<clip_id>/GT/medical_spatiotemporal_vqa/<category>.json
07-stabilized_media_dataset/<clip_id>/GT/medical_spatiotemporal_vqa/<category>.json
```

| Field | Contents |
|---|---|
| `response.answer` | Scene description |
| `review` | Annotation-source phrase flags and caption/rationale length statistics |
| `generation` | Prompt version, input hashes, model configuration, request details, and token usage |

The scene-description response has this form:

```json
{"answer": "Description of the scene and designated subject."}
```

Medical results contain `response.reasoning_process` and `response.answer`.
The runner checks the answer against human GT, checks cited points/slots/times/spans, and checks mandatory evidence coverage.
Accepted records use the original human answer. Answer-only `GT/medical_vqa.json` is not overwritten.
`review.source_reference_check` reports `no_flags` or `flagged` for explicit references to annotation inputs.
This field does not indicate human approval or visual verification. Flags do not reject or rewrite results.
Lengths include total characters and prose word counts, excluding evidence markup from prose counts.
Scene input inspection reports unavailable medical categories under `context_exclusions`; missing answers are not converted to zero.
These checks do not establish visual support for the narrative. Review each task on a pilot before batch generation.
Changes to inputs, prompts, configuration, or generator versions trigger regeneration. A failed request preserves the previous file.

## Prompt files

The [RGB prompt directory](yaml/RGB) contains four task prompts:

| YAML file | Task |
|---|---|
| `native_general_scene_description.yaml` | Describe the native scene and designated subject |
| `stabilized_general_scene_description.yaml` | Describe the subject and visible surroundings in the stabilized crop |
| `native_medical_spatiotemporal_vqa.yaml` | Medical answers with spatial and temporal evidence in native videos |
| `stabilized_medical_spatiotemporal_vqa.yaml` | Medical answers with spatial and temporal evidence in stabilized videos |

`generate.py --tasks all` selects all four prompts. Use `--tasks TASK_NAME ...` to select a subset, or `--task TASK_NAME` for one.
The YAML filename without its extension is the task name.
The [answer-only training workflow](../train/README.md) uses human medical labels directly.

## Generation options

To change the defaults, set these values in `.env`:

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_MODEL` | `gemini/gemini-4-argon` | Gemini model route |
| `GEMINI_VIDEO_FPS` | `source` | Requested source FPS, or a positive numeric FPS |
| `GEMINI_MEDIA_RESOLUTION` | `high` | `low`, `medium`, or `high` |
| `GEMINI_REASONING_EFFORT` | `high` | `low`, `medium`, or `high` |
| `GEMINI_MAX_OUTPUT_TOKENS` | `32768` | Maximum response tokens |
| `COT_CACHE_DIR` | `~/.cache/triage-cot` | Reusable uploaded-file references |

`--video-fps` overrides the FPS configuration.

The gateway route sends inline video with each request. It does not reuse Google file uploads or create explicit provider caches.
The direct Google route reuses active uploads. Both routes skip matching saved results.
Provider-reported cache usage appears in `generation.usage`; caching is not guaranteed.
A missing or null cached-token count means the provider did not report it. It does not mean zero cached tokens.
Files API operations have a combined 300-second deadline and a 60-second request timeout.
Completion requests have a 300-second timeout with up to two configured retries.

## Troubleshooting

| Problem | Action |
|---|---|
| Prepared files differ from provenance | Rerun preprocessing |
| Transcript or viewpoint missing | Complete the source annotation and refresh preprocessing |
| Authentication or model-access error | Check the gateway URL, gateway key, and model route |
| Empty, invalid, or truncated response | Inspect the request and output-token limit, then retry |
