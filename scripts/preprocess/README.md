# Prepare video datasets

[Repository guide](../../README.md) · [Training data](../train/README.md#prepare-the-dataset)

Create native and stabilized training videos with a green box around the designated subject.
The writer copies the human annotations and preserves source resolution and frame timing in lossless H.264 MP4 files.

| Source | Destination |
|---|---|
| `04-segmented_snippets` | `06-native_media_dataset` |
| `05-segmented_stabilized_snippets` | `07-stabilized_media_dataset` |

## Requirements

- Complete the [repository installation](../../README.md#installation), including FFmpeg and FFprobe.
- Mount the source dataset and provide write access to the output directory.
- Set the dataset path in `.env`:

```dotenv
TRIAGE_DATASET_ROOT="/path/to/dataset-root"
```

No API key is needed. Run commands from the repository root.

### Environment variables

| Variable | Used for | Required? |
|---|---|---|
| `TRIAGE_DATASET_ROOT` | Parent directory of `04` and `05` | Set this, or pass `--source-root` |

The command loads the repository `.env` automatically. An existing shell value takes precedence.
`--source-root` overrides the environment value. `--output-root` controls a separate output location.
Gemini, W&B, and cloud credentials are not used by this script.

## Inspect the next run

```bash
uv run python scripts/preprocess/write_media_datasets.py --inspect
```

The report shows which outputs are missing, changed, or unchanged.
Inspection compares metadata, file sizes, and modification times. It does not decode or hash videos.

## Create or update videos

```bash
uv run python scripts/preprocess/write_media_datasets.py --workers 8 --threads-per-worker 1
```

This creates missing outputs, rewrites changed outputs, and retains unchanged outputs.
The writer compares file sizes, modification times, file lists, and overlay configuration with the previous verified output.
Matching clips skip content hashing and video decoding. Changed clips receive content checks; rewritten videos retain full lossless verification.
This fast check does not detect content edits that preserve both file size and modification time.

The default selection includes currently completed minimal/full clips, in both RGB and thermal.
Completion comes from the website's list:

```text
<dataset-root>/00-data_website/medical/completed_clips.csv
```

A completed annotation can still need a generated video. Rerun the writer after new completions or source changes.
Stills and clips absent from the completion list are excluded.

## Other run options

Add these arguments to the generation or inspection command:

| Argument | Effect |
|---|---|
| `--variant native` | Process only `04` → `06` |
| `--variant stabilized` | Process only `05` → `07` |
| `--limit 1` | Check one eligible clip per variant |
| `--write` | Regenerate every selected eligible output, including unchanged ones |
| `--selection all-non-rejected` | Include incomplete exports that have no rejection marker |
| `--source-root /path/to/source` | Override the dataset path in `.env` |
| `--output-root /path/to/output` | Write `06` and `07` under a different directory |
| `--completed-manifest /path/to/completed_clips.csv` | Read a different completion list |
| `--box-px 2` | Set the box outline thickness. Default: 2 pixels |
| `--report /path/to/new-report.json` | Save an additional report to a new file |

`--inspect` and `--write` cannot be combined.
`--limit` includes unchanged clips and validation failures in its count. Native and stabilized selections can differ in limited runs.

The default worker count is one. Set `--workers` to control concurrent clips.
With multiple workers, FFmpeg defaults to one thread per worker. If memory or storage becomes the bottleneck, reduce workers.
For long runs over SSH, use `tmux`. Press Ctrl-C once to cancel queued work and let active workers finish or clean up.

## Read the results

| Report item | Meaning |
|---|---|
| Create / Created | An output is missing / a new output was written |
| Update / Rewritten | An output needs replacement / the replacement was validated and written |
| Skip / Unchanged | An output appears current / it passed generation checks and was retained |
| Verify | Generation must inspect missing inputs or older output metadata |
| Excluded | Selection rules or `--limit` leave the clip out |
| Ineligible | Required eligibility information is missing or invalid |
| Failed | Processing or validation failed |

**Total processed = Created + Rewritten.** Counts describe this run, not all existing output folders.
Each generation run saves `media-writer-run-<id>.json` under the output root, including per-clip failures.

## Output files

Each prepared clip contains its video, source metadata, copied annotations, `provenance.json`, and a `GT/` directory.
See the [dataset layout](../../README.md#data-directories).

A full completed-selection run also publishes `dataset_manifest.json` inside each selected dataset.
It lists successfully prepared clips that remain eligible at publication time.
GT preparation and splitting use these manifests, without consulting the live website list.
Failed or newly completed but unprocessed clips are excluded from the published inventory.
Limited, interrupted, and `all-non-rejected` runs do not replace these manifests.
After such runs, complete a default run before preparing GT or splits.

The writer verifies decoded frames, timestamps, and copied annotations before publishing a replacement.
Existing `GT/` files are preserved. After annotation changes, [refresh the training targets](../train/README.md#prepare-the-dataset).
Outputs for clips removed from the completion list remain on disk but are excluded from the default run.

## Troubleshooting

| Problem | Action |
|---|---|
| Completion CSV missing or malformed | Check the dataset path and the website's completion export |
| Export names a different source video | Refresh the annotation export on the website, then rerun |
| Source changed during processing | Let the source update finish, then rerun |
| Missing metadata, invalid boxes, or decoding error | Review the clip's failure in the JSON report |
| Unexpectedly many existing folders | The prepared manifest determines dataset membership; old folders can remain on disk |
