"""The actual Trainer saves recovery and reports health before generation can fail."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig

from scripts.train.sft.trainer import MedicalSFTTrainer
from scripts.train.shared.runtime import validate_output_directory


class ReliabilityFlowTests(unittest.TestCase):
    def test_generation_failure_preserves_real_recovery_and_health(self):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=8, n_embd=8,
            n_layer=1, n_head=1, bos_token_id=1, eos_token_id=1,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        rows = Dataset.from_list([{'input_ids':[1,2,3,4], 'labels':[-100,-100,3,4],
            'reference':{'variant':'native','category':'injury'}}] * 2)
        def collate(examples):
            return dict(input_ids=torch.tensor([e['input_ids'] for e in examples]),
                        labels=torch.tensor([e['labels'] for e in examples]),
                        attention_mask=torch.ones(len(examples),4,dtype=torch.long),
                        medical_token_groups=torch.tensor([[0,0,1,2]]*len(examples)))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {str(i):i for i in range(16)},unk_token='0')),pad_token='0',eos_token='1')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def failing_evaluator(trainer, dataset, prefix, **kwargs):
                marker=root/'recovery/checkpoint-1/recovery_complete.json'
                self.assertTrue(marker.is_file(), 'Recovery must precede generation')
                self.assertTrue(any('health_loss' in row for row in trainer.state.log_history))
                self.assertIsNone(trainer.state.best_model_checkpoint)
                raise RuntimeError('injected generation failure')
            args=SFTConfig(output_dir=str(root/'checkpoints'),use_cpu=True,bf16=False,
                report_to='none',disable_tqdm=True,max_steps=1,per_device_train_batch_size=1,
                per_device_eval_batch_size=1,eval_strategy='steps',eval_steps=1,
                save_strategy='steps',save_steps=1,load_best_model_at_end=True,
                metric_for_best_model='macro_f1',greater_is_better=True,max_grad_norm=1.,
                dataset_kwargs={'skip_prepare_dataset':True},remove_unused_columns=False,
                label_names=['labels'])
            with patch('scripts.train.sft.trainer.SFTWeightSync'):
                trainer=MedicalSFTTrainer(model=model,args=args,train_dataset=rows,eval_dataset=rows,
                    data_collator=collate,processing_class=tokenizer,
                    medical_evaluator=failing_evaluator,vllm_server_url='http://unused',
                    recovery_policy={'before_evaluation':True,'save_steps':100,'save_total_limit':2},
                    health_policy={'every_n_steps':1,'max_examples':2})
            with self.assertRaisesRegex(RuntimeError,'injected generation failure'):
                trainer.train()
            recovery=root/'recovery/checkpoint-1'
            validate_output_directory(root,recovery)
            for name in ('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth'):
                self.assertTrue((recovery/name).is_file(),name)
            self.assertEqual(json.loads((recovery/'trainer_state.json').read_text())['global_step'],1)
            self.assertIsNone(trainer.state.best_model_checkpoint)
            self.assertEqual(trainer.args.output_dir,str(root/'checkpoints'))
            self.assertFalse((root/'checkpoints/checkpoint-1').exists())


if __name__ == '__main__':
    unittest.main()
