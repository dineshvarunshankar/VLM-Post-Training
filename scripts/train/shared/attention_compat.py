"""Keep CPU export verification independent of optional CUDA attention kernels."""

from contextlib import contextmanager
from functools import wraps
from inspect import unwrap
from threading import RLock


_CPU_EXPORT_LOCK = RLock()


def _cpu_dispatch(kernel):
    reference = unwrap(kernel)

    @wraps(kernel)
    def dispatch(*args, **kwargs):
        query = args[0] if args else kwargs['query'] if 'query' in kwargs else kwargs['hidden_states']
        implementation = reference if query.device.type == 'cpu' else kernel
        return implementation(*args, **kwargs)

    return dispatch


@contextmanager
def cpu_export_attention(model):
    """Use Qwen's reference attention and convolution for CPU export.

    Transformers 5.21 selects installed FLA and convolution kernels even for CPU tensors.
    Its decorated reference functions remain available through ``__wrapped__``.
    Restore the original dispatchers after verification, including on failure.
    CUDA calls retain their original kernels while the context is active.
    """
    if (getattr(model.config, 'model_type', None) not in {'qwen3_5', 'qwen3_5_text'}
            or next(model.parameters()).device.type != 'cpu'):
        yield
        return

    from transformers.models.qwen3_5 import modeling_qwen3_5

    names = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule',
             'causal_conv1d_fn', 'causal_conv1d_update')
    with _CPU_EXPORT_LOCK:
        original = {name: getattr(modeling_qwen3_5, name) for name in names}
        try:
            for name, kernel in original.items():
                setattr(modeling_qwen3_5, name, _cpu_dispatch(kernel))
            yield
        finally:
            for name, kernel in original.items():
                setattr(modeling_qwen3_5, name, kernel)
