"""Separate LoRA merge arithmetic from standalone serialization validation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM
from scripts.train.storage.export_model import export_merged, verify_export


def fixture():
    torch.manual_seed(42)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32)).to(torch.bfloat16)
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM',r=2,
        lora_alpha=2,target_modules=['q_proj'],lora_dropout=0.0))
    with torch.no_grad():
        for name,value in model.named_parameters():
            if 'lora_' in name:value.fill_(0.125)
    class Processor:
        def tokenizer(self,*args,**kwargs):return {'input_ids':torch.tensor([[1,2,3]])}
        def save_pretrained(self,path):pass
    return model,Processor()


class ExportMergeVerificationTest(unittest.TestCase):
    def test_reload_compared_to_merged_and_adapter_drift_reported(self):
        model,processor=fixture()
        # Simulate the observed premerge-only drift without changing saved weights.
        original=model.forward
        def shifted(*args,**kwargs):
            output=original(*args,**kwargs);output.logits=output.logits+1;return output
        model.forward=shifted
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);model.save_pretrained(run/'adapter')
            export_merged(model,processor,run,{})
            report=verify_export(run)
            self.assertTrue(report['merge_weights_verified'])
            self.assertFalse(report['adapter_to_merged_logits']['within_tolerance'])
            self.assertTrue(report['merged_to_reload_logits']['within_tolerance'])

    def test_real_qwen_cpu_export_with_nonzero_adapter(self):
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        config = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=['linear_attention', 'full_attention'], max_position_embeddings=64)
        base = Qwen3_5ForCausalLM(config).to(torch.bfloat16)
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2,
            lora_alpha=2, target_modules=['q_proj'], lora_dropout=0.0))
        with torch.no_grad():
            for name, value in model.named_parameters():
                if 'lora_' in name: value.fill_(0.125)
        _, processor = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); model.save_pretrained(run/'adapter')
            export_merged(model, processor, run, {})
            report = verify_export(run)
            self.assertTrue(report['merge_weights_verified'])
            self.assertEqual(report['merged_to_reload_logits']['max_abs_error'], 0.0)

    def test_corrupt_merge_is_rejected(self):
        model,processor=fixture();original=model.merge_and_unload
        def corrupt(*args,**kwargs):
            merged=original(*args,**kwargs)
            with torch.no_grad():merged.model.layers[0].self_attn.q_proj.weight.add_(1)
            return merged
        model.merge_and_unload=corrupt
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);model.save_pretrained(run/'adapter')
            with self.assertRaisesRegex(ValueError,'Merged weight differs'):
                export_merged(model,processor,run,{})
            self.assertFalse((run/'merged').exists())

    def test_corrupt_reload_is_rejected(self):
        model,processor=fixture();original=LlamaForCausalLM.from_pretrained
        def corrupt(*args,**kwargs):
            reloaded,info=original(*args,**kwargs)
            with torch.no_grad():reloaded.lm_head.weight.add_(10)
            return reloaded,info
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);model.save_pretrained(run/'adapter')
            with patch.object(LlamaForCausalLM,'from_pretrained',side_effect=corrupt):
                with self.assertRaises((AssertionError,ValueError)):
                    export_merged(model,processor,run,{})
            self.assertFalse((run/'merged').exists())

if __name__=='__main__':unittest.main()
