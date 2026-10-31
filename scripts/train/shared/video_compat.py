"""Preserve Gemma video frame counts across Transformers and TRL forwards."""

from contextvars import ContextVar
from functools import wraps
from inspect import signature


def install_video_frame_compatibility(model):
    """Bridge the frame-count omissions in Transformers 5.21 Gemma models.

    Both Gemma outer forwards drop ``num_frames_per_video`` before the backbone.
    Both backbones omit it again when calling ``get_video_features``.
    TRL's fused LM head bypasses the outer forward, so capture both entry points.
    Feature extraction stays in its original position, inside module/FSDP hooks.
    """
    if getattr(model.config, 'model_type', None) not in {'gemma4', 'gemma4_unified'}:
        return
    if getattr(model, '_triage_video_frame_compatibility', False):
        return
    backbone = model.base_model
    current_frames = ContextVar('gemma_video_frames', default=None)

    def capture(original, *, inherit):
        parameters = list(signature(original).parameters)
        position = parameters.index('num_frames_per_video') if 'num_frames_per_video' in parameters else None

        @wraps(original)
        def forward(*args, **kwargs):
            forwarded_frames = kwargs.pop('_triage_video_frames', None) if inherit else None
            if 'num_frames_per_video' in kwargs:
                frames = kwargs['num_frames_per_video']
            elif position is not None and position < len(args):
                frames = args[position]
            elif forwarded_frames is not None:
                frames = forwarded_frames
            else:
                frames = current_frames.get() if inherit else None
            if inherit and position is None:
                # Unified accepts this through **kwargs; do not forward video
                # metadata farther into the language/attention stack.
                kwargs.pop('num_frames_per_video', None)
            if not inherit:
                # The outer HF forward drops the named argument but passes
                # **kwargs through. Keep the exact tensor in backbone inputs so
                # checkpoint replay also works after this context has exited.
                kwargs['_triage_video_frames'] = frames
            token = current_frames.set(frames)
            try:
                return original(*args, **kwargs)
            finally:
                current_frames.reset(token)

        return forward

    original_features = backbone.get_video_features

    @wraps(original_features)
    def video_features(*args, **kwargs):
        # The third positional argument is the explicit per-video frame count.
        if len(args) < 3 and 'num_frames_per_video' not in kwargs:
            kwargs['num_frames_per_video'] = current_frames.get()
        return original_features(*args, **kwargs)

    backbone.get_video_features = video_features
    backbone.forward = capture(backbone.forward, inherit=True)
    model.forward = capture(model.forward, inherit=False)
    model._triage_video_frame_compatibility = True
