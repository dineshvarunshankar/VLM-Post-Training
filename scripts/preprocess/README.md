# Video preprocessing

[Repository guide](../../README.md) · [CoT generation](../cot/README.md) · [Training](../train/README.md)

Create `06` and `07` from the source videos in `04` and `05`.
The writer adds a green subject box and copies the annotations and video sidecar.
Output uses lossless H.264 MP4 with the source resolution and frame timing. The default box thickness is 2 pixels.

## 1. Configure the dataset path

Complete the [installation](../../README.md#installation). Run commands from the repository root.
Set the mounted dataset path in `.env`:

```dotenv
TRIAGE_DATASET_ROOT="/path/to/dataset-root"
```

The source must contain `04-segmented_snippets` and `05-segmented_stabilized_snippets`.
The output must be writable. No API key is required.

## 2. Inspect, then generate

Preview the next run:

```bash
uv run python scripts/preprocess/write_media_datasets.py --inspect
```

Create missing outputs and update changed outputs:

```bash
uv run python scripts/preprocess/write_media_datasets.py
```

| Mode | Action |
|---|---|
| No mode flag | Create missing outputs, update changed outputs, skip unchanged outputs |
| `--inspect` | Show the expected actions without writing videos |
| `--write` | Regenerate every selected eligible clip, including unchanged outputs |

`--inspect` and `--write` cannot be combined. Both native and stabilized videos are selected by default.

## 3. Choose the clips

| Selection | Included clips |
|---|---|
| `--selection completed` (default) | Minimal or full video clips in the current completion CSV |
| `--selection all-non-rejected` | Non-rejected exports, including incomplete clips |

Both selections include RGB and thermal videos. Stills are excluded.

Completed mode reads:

```text
<dataset-root>/00-data_website/medical/completed_clips.csv
```

The site refreshes this list every 10 minutes and at startup.
The writer uses this list because exported completion flags can remain stale after a clip is reopened.
It checks membership again before publishing each new output. A missing or malformed list stops the run.

To include incomplete, non-rejected clips:

```bash
uv run python scripts/preprocess/write_media_datasets.py --selection all-non-rejected --inspect
uv run python scripts/preprocess/write_media_datasets.py --selection all-non-rejected
```

This broader mode uses exported rejection markers. A rejection in either counterpart excludes the pair.
Every selected clip still requires valid metadata, media, and usable subject boxes.

## 4. Control the run

| Argument | Purpose |
|---|---|
| `--variant native` or `--variant stabilized` | Process one variant. Default is both |
| `--limit 1` | Check one eligible clip per variant, including unchanged or failed clips |
| `--source-root /path/to/source` | Override the source dataset path |
| `--output-root /path/to/output` | Write `06` and `07` under another parent directory |
| `--completed-manifest /path/to/completed_clips.csv` | Use another completion CSV |
| `--workers 8` | Process eight clips concurrently. Default is one |
| `--threads-per-worker 1` | Limit FFmpeg threads. Default is one with parallel workers |
| `--box-px 2` | Set the outline thickness |
| `--report /path/to/new-report.json` | Save an additional report. The file must not exist |

A limited run selects clips independently for each variant. It does not guarantee a matching pair.

For a parallel update of both datasets:

```bash
uv run python scripts/preprocess/write_media_datasets.py --workers 8 --threads-per-worker 1
```

Use `--workers "$(nproc)"` to request all available CPU cores.
Reduce the worker count if memory or storage throughput limits the run.
Add `--write` only when every selected output needs regeneration.

For a long SSH session, run the command inside `tmux`.
Press Ctrl-C once to cancel queued work and let active workers finish or clean up.

## 5. Read the report

Inspection counts source MP4 files and shows the expected actions for each variant:

| Column | Meaning |
|---|---|
| Create | No matching output exists |
| Update | An input or writer configuration changed |
| Skip | The output appears unchanged |
| Verify | Generation must check missing inputs or older output metadata |
| Excluded | Selection rules or `--limit` leave the clip out |

The inventory also shows modality and completion totals.
The completion CSV cannot distinguish in-progress, reopened, skipped, and rejected clips outside its list.

Inspection reads metadata and file statistics. It does not decode videos or hash video contents.
Generation performs full checks, so its final actions can differ from the preview.

The final summary reports counts for this run:

| Result | Meaning |
|---|---|
| Created | A new output was written |
| Rewritten | An existing output was replaced after validation |
| Unchanged | An existing valid output was retained |
| Rejected | A rejection rule excluded the clip |
| Ineligible | Required eligibility information is missing or invalid |
| Failed | Processing or validation failed |

**Total processed = Created + Rewritten.** Failures and ineligible reasons appear below the totals.
Each generation run saves `media-writer-run-<id>.json` under the output root.

## Outputs and reruns

| Source | Output |
|---|---|
| `04-segmented_snippets` | `06-native_media_dataset` |
| `05-segmented_stabilized_snippets` | `07-stabilized_media_dataset` |

Each output uses the [per-clip layout](../../README.md#dataset-layout).
The writer checks decoded frames, timestamps, and copied annotations before publishing a clip.
It preserves original filenames, annotation contents, and existing `GT/` files.

Rerun the normal command after source changes. Downstream targets must be refreshed when their input hashes change.
The writer does not delete older outputs that become rejected or disappear from the completion list.

Compatible outputs from the previous flat layout move into clip folders after validation, without unnecessary re-encoding.
Conflicting or incomplete destinations produce errors. Source folders remain unchanged.
