"""CPU export must bypass optional CUDA-only Qwen convolution dispatchers."""
from functools import wraps
from inspect import unwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from scripts.train.shared.attention_compat import cpu_export_attention, _cpu_dispatch

class CPUConvolutionExportTests(unittest.TestCase):
    def test_convolution_prefill_decode_and_restore(self):
        model=SimpleNamespace(config=SimpleNamespace(model_type='qwen3_5'),parameters=lambda:iter([torch.zeros(1)]))
        originals={name:getattr(qwen,name) for name in ('causal_conv1d_fn','causal_conv1d_update')}
        def cuda_only(name):
            @wraps(unwrap(originals[name]))
            def wrapper(*args,**kwargs):
                raise AssertionError('CUDA convolution called with CPU tensors')
            return wrapper
        wrapped={name:cuda_only(name) for name in originals}
        x=torch.randn(1,4,6);weight=torch.randn(4,3);state=torch.randn(1,4,3)
        expected=unwrap(originals['causal_conv1d_fn'])(x,weight,activation='silu')
        expected_update=unwrap(originals['causal_conv1d_update'])(x[:,:,:1],state.clone(),weight,activation='silu')
        with patch.multiple(qwen,**wrapped):
            with cpu_export_attention(model):
                actual=qwen.causal_conv1d_fn(hidden_states=x,weight=weight,activation='silu')
                actual_update=qwen.causal_conv1d_update(x[:,:,:1],state.clone(),weight,activation='silu')
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(actual_update,expected_update)
            for name in wrapped:self.assertIs(getattr(qwen,name),wrapped[name])
            with self.assertRaisesRegex(RuntimeError,'export failed'):
                with cpu_export_attention(model):raise RuntimeError('export failed')
            for name in wrapped:self.assertIs(getattr(qwen,name),wrapped[name])

    def test_cuda_dispatch_keeps_optimized_kernel(self):
        def reference(hidden_states):raise AssertionError('CPU reference used for CUDA input')
        @wraps(reference)
        def kernel(hidden_states):return 'cuda-kernel'
        self.assertEqual(_cpu_dispatch(kernel)(hidden_states=SimpleNamespace(device=SimpleNamespace(type='cuda'))),'cuda-kernel')

if __name__=='__main__':unittest.main()
