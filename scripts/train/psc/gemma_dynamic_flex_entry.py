"""Explicit dynamic FlexAttention graphs with safe checkpoint replay."""
import os
import runpy
import sys
import torch
import transformers

if os.environ.get('TRIAGE_GEMMA_DYNAMIC_FLEX_CHECK') != '1':
    raise SystemExit('Diagnostic requires TRIAGE_GEMMA_DYNAMIC_FLEX_CHECK=1')
if torch.__version__.split('+')[0] != '2.13.0' or transformers.__version__ != '5.21.0':
    raise SystemExit('Revalidate diagnostic for changed torch/transformers versions')
from transformers.integrations import flex_attention as integration

torch._C._dynamo.eval_frame._set_lru_cache(False)
torch._dynamo.config.recompile_limit = 32
from checkpoint_rng_patch import install
install()

@torch.compiler.disable(recursive=False)
def initialize(self, training):
    if not self._is_flex_compiled or training != self.training:
        self.training = training
        self._compiled_flex_attention = torch.compile(integration.flex_attention, dynamic=True, fullgraph=True)
        self._is_flex_compiled = True

integration.WrappedFlexAttention._instance = None
integration.WrappedFlexAttention.__init__ = initialize
print('GEMMA_DYNAMIC_FLEX_REPLAY_ENABLED', flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
