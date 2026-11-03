"""Process-local RNG preservation for FSDP activation replay with nonzero dropout."""
import importlib


def install():
    module = importlib.import_module('torch.distributed.algorithms._checkpoint.checkpoint_wrapper')
    original = module.checkpoint_wrapper
    if getattr(original, '_triage_preserves_rng', False):
        return

    def preserving_wrapper(*args, **kwargs):
        kwargs['preserve_rng_state'] = True
        return original(*args, **kwargs)

    preserving_wrapper._triage_preserves_rng = True
    module.checkpoint_wrapper = preserving_wrapper
