"""Exercise the production Flex wrapper in worker and checkpoint replay contexts."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint
from torch.distributed.algorithms._checkpoint import checkpoint_wrapper as checkpoint_module
from transformers.integrations import flex_attention as integration

ENTRY = Path(__file__).resolve().parents[1] / 'psc/gemma_dynamic_flex_entry.py'


@contextmanager
def production_wrapper(compiled):
    """Replace only GPU compilation and training dispatch; execute the entrypoint."""
    run_path = runpy.run_path
    original_init = integration.WrappedFlexAttention.__init__
    original_instance = integration.WrappedFlexAttention._instance
    with patch.dict('os.environ', {'TRIAGE_GEMMA_DYNAMIC_FLEX_CHECK': '1'}), \
         patch.object(sys, 'argv', [str(ENTRY), 'train.py', '--config', 'fixture.yaml']), \
         patch.object(checkpoint_module, 'checkpoint_wrapper', checkpoint_module.checkpoint_wrapper), \
         patch('runpy.run_path') as dispatch, \
         patch('torch.compile', return_value=compiled) as compile_call, \
         patch('torch._C._dynamo.eval_frame._set_lru_cache'), \
         torch._dynamo.config.patch(recompile_limit=8):
        try:
            run_path(str(ENTRY), run_name='__main__')
            wrapper = integration.WrappedFlexAttention(training=True)
            compile_call.assert_called_once_with(integration.flex_attention, dynamic=True, fullgraph=True)
            dispatch.assert_called_once_with('train.py', run_name='__main__')
            yield wrapper._compiled_flex_attention
        finally:
            integration.WrappedFlexAttention.__init__ = original_init
            integration.WrappedFlexAttention._instance = original_instance


class FlexContextTests(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()

    def test_worker_compiles_more_than_default_limit_and_restores_context(self):
        # Removing the per-call scope reproduces the production limit-8 failure.
        seen = []
        def backend(graph, inputs):
            seen.append(torch._dynamo.config.recompile_limit)
            return graph.forward
        def calculation(x):
            return x.sin() + 1
        compiled = torch.compile(calculation, backend=backend, fullgraph=True, dynamic=False)
        with production_wrapper(compiled) as scoped:
            def worker():
                before = torch._dynamo.config.recompile_limit
                for size in range(2, 12):
                    x = torch.ones(size)
                    torch.testing.assert_close(scoped(x), calculation(x))
                return before, torch._dynamo.config.recompile_limit
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(worker).result(), (8, 8))
        self.assertEqual(seen, [32] * 10)

    def test_checkpoint_replay_uses_calling_context_and_preserves_gradients(self):
        seen = []
        def calculation(x):
            seen.append(torch._dynamo.config.recompile_limit)
            return x.sin() * x
        with production_wrapper(calculation) as scoped:
            x = torch.tensor([.2, .7], requires_grad=True)
            y = checkpoint(scoped, x, use_reentrant=False)
            # A separate context models backward replay's compiler configuration.
            with torch._dynamo.config.patch(recompile_limit=8):
                y.sum().backward()
                self.assertEqual(torch._dynamo.config.recompile_limit, 8)
            expected = x.detach().sin() + x.detach() * x.detach().cos()
            torch.testing.assert_close(x.grad, expected)
        self.assertGreaterEqual(len(seen), 2)
        self.assertEqual(set(seen), {32})

    def test_error_propagates_and_restores_calling_context(self):
        seen = []
        def fail(x):
            seen.append(torch._dynamo.config.recompile_limit)
            raise RuntimeError('compiler failure remains visible')
        with production_wrapper(fail) as scoped:
            with torch._dynamo.config.patch(recompile_limit=11):
                with self.assertRaisesRegex(RuntimeError, 'compiler failure remains visible'):
                    scoped(torch.ones(1))
                self.assertEqual(torch._dynamo.config.recompile_limit, 11)
        self.assertEqual(seen, [32])


if __name__ == '__main__':
    unittest.main()
