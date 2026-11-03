"""Validation loss and medical metrics share one evaluation event."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from datasets import Dataset
from trl import SFTConfig
from scripts.train.sft.trainer import MedicalSFTTrainer


class ValidationLossTests(unittest.TestCase):
    def test_real_loss_only_loop_with_masked_labels_and_test_prefix(self):
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=16,
            n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=1, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        rows = Dataset.from_list([{'input_ids': [1, 2, 3, 4],
                                  'labels': [-100, -100, 3, 4]}] * 3)
        def collate(examples):
            batch = {key: torch.tensor([row[key] for row in examples])
                     for key in ('input_ids', 'labels')}
            batch['medical_token_groups'] = torch.tensor([[0, 0, 1, 2]] * len(examples))
            batch['attention_mask'] = torch.ones_like(batch['input_ids'])
            return batch
        # The repository uses an external multimodal processor; these tests use
        # a minimal tokenizer object and already prepared examples instead.
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
            WordLevel({str(i): i for i in range(16)}, unk_token='0')),
            pad_token='0', eos_token='1')
        with tempfile.TemporaryDirectory() as output:
            args = SFTConfig(output_dir=output, use_cpu=True, bf16=False,
                report_to='none', per_device_eval_batch_size=1,
                dataset_kwargs={'skip_prepare_dataset': True},
                remove_unused_columns=False, label_names=['labels'])
            captured = []
            def evaluator(trainer, dataset, prefix, *, additional_metrics=None):
                metrics = dict(additional_metrics or {}, **{prefix+'_macro_f1': .4})
                captured.append(metrics)
                return metrics
            with patch('scripts.train.sft.trainer.SFTWeightSync'):
                trainer = MedicalSFTTrainer(model=model, args=args,
                    train_dataset=rows, eval_dataset=rows, data_collator=collate,
                    processing_class=tokenizer, medical_evaluator=evaluator,
                    vllm_server_url='http://unused')
            model.eval()
            with torch.no_grad():
                inputs = collate([rows[0]])
                inputs.pop('medical_token_groups')
                outputs = model(**inputs)
                expected = outputs.loss.item()
                value_loss = torch.nn.functional.cross_entropy(outputs.logits[:, 1], torch.tensor([3])).item()
                value_correct = float(outputs.logits[0, 1].argmax() == 3)
            model.train()
            # The diagnostic metadata must never alter training loss or reach
            # the model as an unexpected keyword argument.
            with_metadata = trainer.compute_loss(model, collate([rows[0]]))
            without_metadata = collate([rows[0]])
            without_metadata.pop('medical_token_groups')
            baseline = trainer.compute_loss(model, without_metadata)
            torch.testing.assert_close(with_metadata, baseline)
            for prefix in ('eval', 'test'):
                result = trainer.evaluate(metric_key_prefix=prefix)
                self.assertIn(prefix+'_loss', result)
                self.assertAlmostEqual(result[prefix+'_loss'], expected, places=5)
                self.assertEqual(result[prefix+'_macro_f1'], .4)
                self.assertEqual(result[prefix+'_medical_value_tokens'], 3)
                self.assertEqual(result[prefix+'_medical_format_tokens'], 3)
                self.assertAlmostEqual(result[prefix+'_medical_value_loss'], value_loss, places=5)
                self.assertEqual(result[prefix+'_medical_value_accuracy'], value_correct)
                self.assertTrue(model.training)
            self.assertEqual(len(captured), 2)

    def test_medical_evaluator_logs_combined_metrics_once(self):
        from scripts.train.shared.evaluation import MedicalVLLMEvaluator
        evaluator = MedicalVLLMEvaluator(None, {}, None,
            {'categories': {'medical': {}}}, {}, 'test-run')
        evaluator.auxiliary = SimpleNamespace(prepare=lambda dataset, prefix: ([], {}), scoring={})
        evaluator.predict_many = lambda trainer, rows, *args: ['{}'] * len(rows)
        rows = [{'reference': {'category': 'medical'}, 'answer': '{}'}]
        trainer = SimpleNamespace(
            vllm_generation=SimpleNamespace(sync_weights=lambda: None),
            state=SimpleNamespace(global_step=1), args=SimpleNamespace(seed=42),
            accelerator=SimpleNamespace(process_index=0, num_processes=1,
                                        wait_for_everyone=lambda: None),
            is_world_process_zero=lambda: False, log=Mock(), control=object(),
            callback_handler=SimpleNamespace(on_evaluate=Mock(return_value='updated')))
        with patch('scripts.train.shared.evaluation.answer_schema', return_value={}), \
             patch('scripts.train.shared.evaluation.gather_object', side_effect=lambda x: x), \
             patch('scripts.train.shared.evaluation.score_medical_evaluation',
                   return_value={'metrics': {'macro_f1': .4}}), \
             patch('scripts.train.shared.evaluation.score_auxiliary',
                   return_value={'tasks': {}, 'sources': {}}):
            metrics = evaluator(trainer, rows, 'eval',
                additional_metrics={'eval_loss': 1.25, 'eval_loss_runtime': .5})
        self.assertEqual(metrics['eval_loss'], 1.25)
        self.assertEqual(metrics['eval_macro_f1'], .4)
        trainer.log.assert_called_once_with(metrics)
        trainer.callback_handler.on_evaluate.assert_called_once()
        self.assertEqual(trainer.callback_handler.on_evaluate.call_args.args[-1], metrics)
        self.assertEqual(trainer.control, 'updated')

    def test_loss_failure_does_not_report_successful_medical_evaluation(self):
        trainer = object.__new__(MedicalSFTTrainer)
        trainer.eval_dataset = [1]
        trainer.model = torch.nn.Linear(1, 1)
        trainer.get_eval_dataloader = Mock(return_value=[])
        trainer.evaluation_loop = Mock(side_effect=RuntimeError('loss failed'))
        trainer.medical_evaluator = Mock(return_value={})
        with self.assertRaisesRegex(RuntimeError, 'loss failed'):
            trainer.evaluate()
        trainer.medical_evaluator.assert_not_called()
        self.assertTrue(trainer.model.training)


if __name__ == '__main__':
    unittest.main()
