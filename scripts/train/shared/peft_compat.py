"""Compatibility for PEFT 0.21.2 and PyTorch activation-checkpoint wrappers."""

from functools import wraps


def install_checkpoint_adapter_compatibility():
    """Match PEFT structural prefixes to PyTorch's canonical state-dict names.

    PEFT 0.21.2 derives adapter prefixes from named_modules(), which includes
    _checkpoint_wrapped_module. PyTorch strips that segment from state_dict().
    Normalize the shared prefix collector so both save and restore see all layers.
    """
    import peft
    from peft.tuners import tuners_utils

    original = tuners_utils._get_tuner_state_dict_key_prefixes
    if getattr(original, '_triage_checkpoint_names', False):
        return
    if peft.__version__ != '0.21.2':
        raise RuntimeError('Revalidate checkpoint adapter compatibility after changing PEFT 0.21.2')

    @wraps(original)
    def canonical_prefixes(*args, **kwargs):
        return {
            '.'.join(part for part in prefix.split('.') if part != '_checkpoint_wrapped_module')
            for prefix in original(*args, **kwargs)
        }

    canonical_prefixes._triage_checkpoint_names = True
    tuners_utils._get_tuner_state_dict_key_prefixes = canonical_prefixes


def validate_adapter_state(model, weights):
    """Reject missing, extra, or malformed adapter tensors before export."""
    from peft import get_peft_model_state_dict

    expected = get_peft_model_state_dict(model, adapter_name='default')
    missing, extra = set(expected) - set(weights), set(weights) - set(expected)
    mismatched = [key for key in set(expected) & set(weights)
                  if expected[key].shape != weights[key].shape]
    if not expected or missing or extra or mismatched:
        raise ValueError(f'Incomplete adapter: expected={len(expected)}, saved={len(weights)}, '
                         f'missing={len(missing)}, extra={len(extra)}, shape_mismatch={len(mismatched)}')
    return len(expected)
