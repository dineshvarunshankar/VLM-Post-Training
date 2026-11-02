"""Real CPU Trainer loaders must keep persistent teacher subsets separate."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import SFTConfig

from scripts.train.sft.trainer import MedicalSFTTrainer


def collate(examples):
    return {key:torch.tensor([row[key] for row in examples]) for key in ('input_ids','labels')}


def teacher(task, identity):
    return {'input_ids':[1,2,3,4],'labels':[-100,-100,3,4],
            'task':task,'identity':identity,'teacher_available':True,
            'reasoning':'Visible evidence.' if task=='medical_vqa' else None}


class TeacherLoaderTests(unittest.TestCase):
    def test_persistent_loaders_keep_medical_caption_validation_and_test_identity(self):
        model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_positions=16,n_embd=8,
            n_layer=1,n_head=1,bos_token_id=1,eos_token_id=1,loss_type='ForCausalLM'))
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {str(i):i for i in range(16)},unk_token='0')),pad_token='0',eos_token='1')
        validation=Dataset.from_list([teacher('medical_vqa','validation-medical')])
        test=Dataset.from_list([teacher('medical_vqa','test-medical')])
        class Evaluator:
            def teacher_caption_rows(self,prefix):
                return [teacher('scene_description','test-caption' if prefix=='test' else 'validation-caption')]
            def __call__(self,trainer,dataset,prefix,additional_metrics=None):
                return dict(additional_metrics or {},**{prefix+'_macro_f1':1.0})
        with tempfile.TemporaryDirectory() as output:
            args=SFTConfig(output_dir=output,use_cpu=True,bf16=False,report_to='none',
                per_device_eval_batch_size=1,dataloader_num_workers=1,
                dataloader_persistent_workers=True,dataloader_prefetch_factor=1,
                dataset_kwargs={'skip_prepare_dataset':True},
                remove_unused_columns=False,label_names=['labels'])
            with patch('scripts.train.sft.trainer.SFTWeightSync'):
                trainer=MedicalSFTTrainer(model=model,args=args,train_dataset=validation,
                    eval_dataset=validation,data_collator=collate,processing_class=tokenizer,
                    medical_evaluator=Evaluator(),vllm_server_url='http://unused')
            trainer.video_collator=SimpleNamespace(answer_format='cot')
            observed=[]
            def inspect_loss_loader(dataloader,*,metric_key_prefix,**kwargs):
                # Inspect real Accelerate-prepared loaders without starting workers.
                observed.append(dataloader.dataset[0]['identity'])
                return SimpleNamespace(metrics={metric_key_prefix+'_loss':1.0})
            trainer.evaluation_loop=inspect_loss_loader
            trainer.evaluate(metric_key_prefix='eval')
            trainer.evaluate(eval_dataset=test,metric_key_prefix='test')
            self.assertEqual(observed,['validation-medical','validation-caption','test-medical','test-caption'])
            replacement=Dataset.from_list([teacher('medical_vqa','replacement-medical')])
            loader=trainer.get_eval_dataloader(replacement)
            self.assertEqual(loader.dataset[0]['identity'],'replacement-medical')
            self.assertTrue(loader.persistent_workers)
            self.assertEqual(loader.num_workers,1)
            same_rows=Dataset.from_list([teacher('medical_vqa','replacement-medical')])
            self.assertIs(trainer.get_eval_dataloader(same_rows),loader)
            self.assertEqual(trainer.train_dataset[0]['identity'],'validation-medical')


if __name__=='__main__': unittest.main()
