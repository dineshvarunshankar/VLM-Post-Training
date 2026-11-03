"""Measure the exact training sampler's frame bound from the selected inventory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

from scripts.train.shared.data import DATASETS, atomic_json


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def measure_frame_bound(plan):
    """Read inventory/annotation bytes only; never decode or change dataset contents."""
    from transformers.video_utils import VideoMetadata
    from scripts.train.shared.video import sample_capped_frames, source_video_config

    config = plan['run']
    video = plan['experiment']['video']
    training_sources = config['dataset']['sources']
    evaluation_sources = plan['experiment'].get('evaluation', {}).get('sources', training_sources)
    split_sources = {'train': training_sources, 'validation': evaluation_sources, 'test': evaluation_sources}
    sources = list(dict.fromkeys([*training_sources, *evaluation_sources]))
    media = Path(config['paths']['media_root'])
    splits = Path(config['paths']['split_dir'])
    caps = {source: source_video_config(video, source)['fps'] for source in sources}
    for source, fps in caps.items():
        if source not in DATASETS or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid sampling source/FPS: {source!r}: {fps!r}')
    clips, hashes = set(), {}
    for split in ('train', 'validation', 'test'):
        path = splits / (split + '.jsonl')
        try:
            raw = path.read_bytes()
            hashes[split] = hashlib.sha256(raw).hexdigest()
            for line_number, line in enumerate(raw.decode().splitlines(), 1):
                row = json.loads(line)
                if row['variant'] in split_sources[split]:
                    clip = row['clip']
                    if not isinstance(clip, str) or not clip or Path(clip).is_absolute() or '..' in Path(clip).parts:
                        raise ValueError(f'Invalid clip at line {line_number}')
                    clips.add((row['variant'], clip))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Cannot measure frame bound from split {path}: {error}') from error
    for source in sources:
        if not any(variant == source for variant, _ in clips):
            raise ValueError(f'No selected inventory clips for source {source!r} in {splits}')
    if not clips:
        raise ValueError(f'No selected inventory clips in {splits}')

    def measure(item):
        variant, clip = item
        path = media / DATASETS[variant] / clip / 'all_annotations/meta.json'
        try:
            raw = path.read_bytes()
            metadata = json.loads(raw)
            frames, fps = metadata['n_frames'], metadata['fps']
            if type(frames) is not int or frames <= 0:
                raise ValueError('n_frames must be a positive integer')
            if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
                raise ValueError('fps must be a positive finite number')
            count = len(sample_capped_frames(VideoMetadata(total_num_frames=frames, fps=fps),
                                             fps_cap=caps[variant]))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Cannot measure frame bound from metadata {path}: {error}') from error
        return {'variant': variant, 'clip': clip, 'sampled_frames': count, 'n_frames': frames,
                'fps': fps, 'meta_sha256': hashlib.sha256(raw).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(measure, sorted(clips)))
    return {
        'profile_video_frames': max(2, max(row['sampled_frames'] for row in rows)),
        'clips': len(rows),
        'split_sha256': hashes,
        'maximum_cases': sorted(rows, key=lambda row: row['sampled_frames'], reverse=True)[:8],
        'source_maxima': {source: max(row['sampled_frames'] for row in rows if row['variant'] == source)
                          for source in sources},
        'method': 'actual sample_capped_frames from annotation FPS/frame-count; no video decoding or truncation',
        'metadata_sha256': _fingerprint(rows),
        'split_sources': split_sources,
        'sampling_sha256': _fingerprint({'split_sources': split_sources, 'video': video}),
    }


def main(argv=None):
    from scripts.train.shared.run_config import load_run, compile_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        report = measure_frame_bound(compile_run(load_run(args.config)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    atomic_json(Path(args.output), report)
    print(json.dumps(report), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
