# CoT annotation generation

[Repository guide](../../README.md) · [Preprocessing](../preprocess/README.md) · [Training](../train/README.md)

Generate annotations from prepared videos with Gemini through LiteLLM.
The current runner generates native RGB scene descriptions. These contain an `answer`, without a separate reasoning field.

## Supported tasks

| Prompt | Runner support |
|---|---|
| `native_general_scene_description` | Implemented for RGB |
| `stabilized_general_scene_description` | Template only |
| `native_medical_spatiotemporal_vqa` | Template only |
| `stabilized_medical_spatiotemporal_vqa` | Template only |

Prompts are in [yaml/RGB](yaml/RGB). The runner selects the native scene task automatically.
Medical templates retain `reasoning_process` and `answer`. CoT dataset assembly is not implemented.
Answer-only training uses a [separate workflow](../train/README.md).

## 1. Configure Gemini

Complete the [installation](../../README.md#installation) and [preprocessing](../preprocess/README.md).
Run commands from the repository root. Set these values in the ignored `.env`:

```dotenv
TRIAGE_DATASET_ROOT="/path/to/dataset-root"
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini/gemini-4-argon
```

Replace the example key with your actual Gemini key. Select a model route available to your account.
The current Files API client uses a direct Gemini key. A gateway-only key is not sufficient for this upload path.

| Variable | Default | Purpose |
|---|---|---|
| `TRIAGE_DATASET_ROOT` | Required | Parent directory containing `06` and `07` |
| `AIRLAB_DATASET_ROOT` | Unset | Optional dataset path. Takes precedence over `TRIAGE_DATASET_ROOT` |
| `GEMINI_API_KEY` | Required for generation | API authentication |
| `GEMINI_MODEL` | `gemini/gemini-4-argon` | LiteLLM route. Must start with `gemini/` |
| `GEMINI_VIDEO_FPS` | `source` | Source FPS or a positive numeric FPS |
| `GEMINI_MEDIA_RESOLUTION` | `high` | `low`, `medium`, or `high` |
| `GEMINI_REASONING_EFFORT` | `high` | `low`, `medium`, or `high` |
| `GEMINI_MAX_OUTPUT_TOKENS` | `8192` | Output-token limit |
| `COT_CACHE_DIR` | `~/.cache/triage-cot` | Cached uploaded-file references |

`--dataset-root` overrides both dataset variables. `--video-fps` overrides the FPS variable.

## 2. Review what the model receives

The runner reads `06-native_media_dataset/<clip_id>/`:

| Input | Data sent or used |
|---|---|
| `<clip_id>.mp4` | Prepared video with the subject overlay |
| `all_annotations/meta.json` | Modality, duration, time of day, and human scene transcription |
| `all_annotations/camera_motion.json` | Camera viewpoint |
| `all_annotations/counting.json` | Recorded clip-wide totals and subject type |
| `provenance.json` | Video and annotation integrity checks |

Counts describe recorded clip totals, not people visible simultaneously.
FPS controls video sampling. Frame dimensions are not included in the scene-description text input.
This task does not send medical evidence points or cue boxes. The subject box is already drawn on the video.

The runner requires RGB, a transcript, a viewpoint, and copied completion status `completed_minimal` or `completed`.
A copied rejection marker excludes the clip. It does not consult the live completion CSV.
Refresh preprocessing after source changes and review the current completion list before selecting clips.

Inspect one input without an API call:

```bash
uv run python scripts/cot/generate.py --limit 1 --inspect
```

For a browser preview, start the server:

```bash
uv run python scripts/cot/preview.py --port 8765
```

On your computer, forward the port:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

Open **http://localhost:8765**. Alternatively, forward port `8765` through VS Code Remote SSH.
The preview shows the video, prompts, input data, generation configuration, and output schema.
It binds to localhost and does not generate annotations or upload media.

All four tasks are available in the preview. Only native scene description has assembled clip inputs. The other tasks show templates.
Stabilized templates do not receive counting annotations.

## 3. Generate and review one clip

Replace `CLIP_ID` with a folder name from `06-native_media_dataset`:

```bash
uv run python scripts/cot/generate.py --clip CLIP_ID --generate
```

`--generate` uploads the MP4, makes a paid API request, and saves the response.
The runner does not convert the video. Review the result before a larger batch.

## 4. Generate a batch

```bash
uv run python scripts/cot/generate.py --limit 10 --generate --workers 4
```

Omit `--limit` to process all discovered native RGB clips. Explicitly selected thermal clips are rejected.
The default is four workers. Actual concurrency is limited by available CPUs and the selected clip count.
Use `--workers 1` for sequential processing.

A failed clip does not stop other clips. The command exits nonzero if any clip fails.
On interruption, queued work is cancelled. Active requests finish or reach their deadlines.

## Outputs and reruns

Results go to:

```text
06-native_media_dataset/<clip_id>/GT/general_scene_description.json
```

| Field | Contents |
|---|---|
| `response.answer` | Generated scene description |
| `generation` | Prompt version, input hashes, configuration, library versions, request ID, timestamp, and token usage |

Matching valid results are skipped. Changes to inputs, prompts, configuration, implementation, or recorded SDK versions trigger regeneration.
Failed requests preserve the previous result. A preserved result can be stale if its inputs changed.

The runner checks JSON format and requires a nonempty answer.
These checks do not establish visual accuracy. Review generated claims before using them for training.

## Caching and timeouts

The runner reuses active uploaded-file references. It places shared instructions and media before clip-specific annotation text.
It does not create explicit context caches. Any implicit cache savings depend on the provider and request.
Returned usage details are saved in `generation.usage`.

Google's SDK uploads files. LiteLLM sends completion requests.
File requests have a 60-second timeout, within a shared 300-second upload-and-polling deadline.
Completion requests have a 300-second timeout and up to two configured retries.
More workers increase API traffic, not the account quota.

## Common problems

| Problem | Action |
|---|---|
| Prepared dataset missing | Run preprocessing or correct the dataset path |
| Files differ from provenance | Refresh preprocessing. Do not edit copied files |
| Transcript or viewpoint missing | Complete source annotations, then refresh preprocessing |
| Authentication fails | Check the direct Gemini key and model access |
| Empty, truncated, or invalid response | Inspect the request and output-token limit, then retry |
