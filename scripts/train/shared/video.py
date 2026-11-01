"""Training video sampling and model-specific spatial budgets."""

import math
from copy import deepcopy
from contextlib import contextmanager
from functools import partial

import numpy as np
from transformers.video_utils import default_sample_indices_fn


def sample_capped_frames(metadata, *, fps_cap, required_frames=(), **kwargs):
    source_fps = metadata.fps
    if source_fps is None or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Video metadata must contain a positive source FPS")
    if metadata.total_num_frames <= 0:
        raise ValueError("Video has no frames")
    indices = default_sample_indices_fn(
        metadata, fps=None if source_fps <= fps_cap else fps_cap
    )
    required = set(required_frames)
    if any(
        type(i) is not int or not 0 <= i < metadata.total_num_frames for i in required
    ):
        raise ValueError("Required evidence frame is outside the video")
    if len(required) > len(indices):
        raise ValueError("Required evidence frames exceed the configured frame budget")
    selected = set(int(i) for i in indices)
    for frame in sorted(required - selected):
        replaceable = selected - required
        nearest = min(replaceable, key=lambda i: (abs(i - frame), i))
        selected.remove(nearest)
        selected.add(frame)
    return np.array(sorted(selected), dtype=int)


def source_video_config(config, source):
    result = deepcopy(config)
    budgets = result.pop("sources", None)
    if budgets is not None:
        if source not in budgets:
            raise ValueError(f"No video budget for source {source!r}")
        selected = budgets[source]
        result["fps"] = selected["fps"]
        result["gemma"]["max_soft_tokens"] = selected["max_soft_tokens"]
        result["qwen"]["max_pixels_per_frame"] = selected["max_pixels_per_frame"]
    return result


def spatial_options(processor, config, *, sampled_frames=None):
    video = processor.video_processor
    if type(video).__name__.startswith("Qwen3VL"):
        size = dict(video.size)
        size["longest_edge"] = config["qwen"]["max_pixels"]
        per_frame = config["qwen"].get("max_pixels_per_frame")
        if per_frame is not None:
            if type(sampled_frames) is not int or sampled_frames < 1:
                raise ValueError("A per-frame Qwen budget requires the sampled frame count")
            # Qwen smart_resize compares against the temporally rounded frame count.
            temporal = getattr(video, "temporal_patch_size", 2)
            frames = max(temporal, round(sampled_frames / temporal) * temporal)
            size["longest_edge"] = min(size["longest_edge"], per_frame * frames)
        return {"videos_kwargs": {"size": size, "cap_pixels_per_frame": False}}
    if type(video).__name__.startswith("Gemma4"):
        return {
            "videos_kwargs": {"max_soft_tokens": config["gemma"]["max_soft_tokens"]}
        }
    raise ValueError(f"No agreed video budget for {type(video).__name__}")


@contextmanager
def capped_sampling(
    processor, fps_cap, required_frames_by_video=None, expected_metadata_by_video=None
):
    video = processor.video_processor
    previous = video.__dict__.get("sample_frames")
    had_override = "sample_frames" in video.__dict__
    pending = (
        iter(required_frames_by_video) if required_frames_by_video is not None else None
    )

    expected = (
        iter(expected_metadata_by_video)
        if expected_metadata_by_video is not None
        else None
    )

    def sample(metadata, **kwargs):
        try:
            required = next(pending) if pending is not None else ()
        except StopIteration as error:
            raise ValueError(
                "Processor sampled more videos than the supplied frame requirements"
            ) from error
        if expected is not None:
            reference = next(expected)
            if reference is not None and (
                metadata.total_num_frames != reference["n_frames"]
                or not math.isclose(
                    metadata.fps, reference["fps"], rel_tol=0, abs_tol=0.001
                )
            ):
                raise ValueError(
                    "Decoded video FPS/frame count differs from annotation metadata"
                )
        return sample_capped_frames(
            metadata, fps_cap=fps_cap, required_frames=required, **kwargs
        )

    video.sample_frames = (
        sample
        if pending is not None
        else partial(sample_capped_frames, fps_cap=fps_cap)
    )
    try:
        yield
        if pending is not None and next(pending, None) is not None:
            raise ValueError(
                "Processor did not consume every video's frame requirements"
            )
    finally:
        if had_override:
            video.sample_frames = previous
        else:
            del video.sample_frames
