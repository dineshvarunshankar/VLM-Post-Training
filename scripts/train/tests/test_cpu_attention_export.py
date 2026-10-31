"""CPU export must work when optional CUDA attention kernels are installed."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from scripts.train.shared.attention_compat import cpu_export_attention
from scripts.train.storage.export_model import export_merged, verify_export


def tiny_qwen():
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['linear_attention', 'full_attention'],
        max_position_embeddings=64,
    )
    return Qwen3_5ForCausalLM(config).to(torch.bfloat16).eval()


class CPUAttentionExportTests(unittest.TestCase):
    def test_cpu_prefill_and_cached_decode_restore_original_kernels(self):
        model = tiny_qwen()
        names = ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule')
        original = {name: getattr(modeling_qwen3_5, name) for name in names}
        with torch.inference_mode(), cpu_export_attention(model):
            prefill = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=True)
            decoded = model(input_ids=torch.tensor([[5]]),
                            past_key_values=prefill.past_key_values, use_cache=True)
        self.assertTrue(torch.isfinite(decoded.logits).all())
        for name, kernel in original.items():
            self.assertIs(getattr(modeling_qwen3_5, name), kernel)
        with self.assertRaisesRegex(RuntimeError, 'verification failed'):
            with cpu_export_attention(model):
                raise RuntimeError('verification failed')
        for name, kernel in original.items():
            self.assertIs(getattr(modeling_qwen3_5, name), kernel)

    @unittest.skipUnless(importlib.util.find_spec('fla'), 'requires the pinned fla-core dependency')
    def test_real_qwen_lora_export_reloads_on_cpu_with_fla_installed(self):
        torch.manual_seed(42)
        model = get_peft_model(tiny_qwen(), LoraConfig(
            task_type='CAUSAL_LM', r=2, lora_alpha=2, target_modules=['q_proj'],
        ))
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'lora_B' in name:
                    value.fill_(0.01)
        backend = Tokenizer(WordLevel({'[UNK]': 0, 'Return': 1, 'JSON': 2, '.': 3}, unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        processor = SimpleNamespace(tokenizer=tokenizer, save_pretrained=tokenizer.save_pretrained)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model.save_pretrained(run / 'adapter')
            export_merged(model, processor, run, {'best_checkpoint': 'tiny-cpu-regression'})
            report = verify_export(run)
            self.assertTrue(report['reload_verified'])
            self.assertTrue(report['adapter_completeness_verified'])
            self.assertEqual(report['adapter_tensor_count'], 2)


if __name__ == '__main__':
    unittest.main()
