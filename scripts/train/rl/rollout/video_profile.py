"""Derive vLLM video profiling flags from the training processor; never launch."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path


def profiling_flags(processor, video_config, *, profile_video_frames, model_config):
    """Keep input processing unchanged; size the startup dummy separately."""
    from scripts.train.shared.video import spatial_options

    if type(profile_video_frames) is not int or profile_video_frames < 2:
        raise ValueError('profile_video_frames must be an integer of at least two')
    kwargs = spatial_options(processor, video_config)
    video = processor.video_processor
    limits = {'image': 0, 'video': {'count': 1}}
    if getattr(model_config, 'audio_config', None) is not None:
        limits['audio'] = 0
    if type(video).__name__.startswith('Gemma4'):
        tokens = video_config['gemma']['max_soft_tokens']
        height = math.isqrt(tokens)
        while tokens % height:
            height -= 1
        width = tokens // height
        factor = video.patch_size * video.pooling_kernel_size
        limits['video'].update(num_frames=profile_video_frames,
                               width=width * factor, height=height * factor)
    # Qwen's profiler derives the whole-video bound from size.longest_edge.
    # A frame/geometry override would shrink its conservative profiling dummy.
    return {'mm_processor_kwargs': kwargs, 'limit_mm_per_prompt': limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--profile-video-frames', type=int, required=True,
                        help='Startup profiling bound covering the dataset; does not truncate input videos')
    parser.add_argument('--field', choices=['all', 'mm_processor_kwargs', 'limit_mm_per_prompt'], default='all')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from scripts.train.shared.run_config import load_run, compile_run
    from scripts.train.shared.modeling import build_processor

    config = load_run(args.run)
    experiment = compile_run(config)['experiment']
    profile = deepcopy(config['model_settings'])
    profile['local_path'] = str(Path(config['paths']['weights_root']) / profile['repository'] / profile['revision'])
    checkpoint_config, processor, _ = build_processor(profile, experiment)
    result = profiling_flags(processor, experiment['video'], profile_video_frames=args.profile_video_frames,
                             model_config=checkpoint_config)
    print(json.dumps(result if args.field == 'all' else result[args.field], sort_keys=True))


if __name__ == '__main__':
    main()
