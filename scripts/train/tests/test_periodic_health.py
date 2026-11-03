"""Periodic teacher diagnostics must not change training or epoch selection."""
import importlib
import math
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch


class PeriodicHealthTests(unittest.TestCase):
    def mixin(self):
        try:
            module = importlib.import_module('scripts.train.sft.health')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'Periodic health diagnostics are not implemented')
        return module.PeriodicHealthMixin

    def trainer(self, rows=None, cot=False):
        class Base:
            def _maybe_log_save_evaluate(self, *args, **kwargs):
                self.epoch_hook_calls += 1
                return 'epoch preserved'
        class Trainer(self.mixin(), Base):
            pass
        trainer = Trainer()
        trainer.eval_dataset = rows if rows is not None else [
            {'reference': {'variant': 'native', 'category': 'injury'}}]
        trainer.video_collator = SimpleNamespace(answer_format='cot' if cot else 'answer_only')
        trainer.model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
        trainer.model[1].eval()
        trainer.state = SimpleNamespace(global_step=0, best_metric=.8, best_model_checkpoint='epoch-1')
        trainer.control = SimpleNamespace(should_training_stop=False, should_save=True, should_evaluate=True)
        trainer.callback_handler = SimpleNamespace(callbacks=[], eval_dataloader='epoch loader')
        trainer.optimizer = SimpleNamespace(eval=Mock(side_effect=AssertionError('Optimizer changed')))
        trainer.epoch_hook_calls = 0
        trainer._metrics = {'train': {'loss': [2.]}, 'eval': {'entropy': [3.]}}
        trainer._eval_dataloaders = {'eval': 'full validation loader'}
        trainer.get_eval_dataloader = lambda dataset: dataset
        trainer.logs = []
        trainer.log = lambda metrics: trainer.logs.append(dict(metrics))
        trainer._teacher_loss_loop = lambda *args, **kwargs: SimpleNamespace(metrics={
            'health_loss': 2., 'health_medical_value_loss': 3.,
            'health_medical_value_accuracy': .25, 'health_medical_value_tokens': 1})
        return trainer

    def test_disabled_and_step_cadence_leave_epoch_hook_in_place(self):
        trainer = self.trainer()
        trainer.configure_health(None)
        trainer.state.global_step = 2
        self.assertEqual(trainer._maybe_log_save_evaluate(), 'epoch preserved')
        self.assertFalse(trainer.logs)
        trainer.configure_health({'every_n_steps': 2, 'max_examples': 1})
        for step in (0, 1, 2, 2, 3, 4):
            trainer.state.global_step = step
            trainer._maybe_log_save_evaluate()
        self.assertEqual(len(trainer.logs), 2)
        self.assertEqual(trainer.epoch_hook_calls, 7)
        self.assertEqual(trainer.state.best_metric, .8)
        self.assertEqual(trainer.state.best_model_checkpoint, 'epoch-1')

    def test_stratified_fixed_subset_and_honest_cot_coverage(self):
        rows = [dict(task='medical_vqa', reference=dict(variant=source, category=category),
                     teacher_available=True, reasoning='validated reasoning')
                for source, category in [('native', 'a'), ('native', 'a'),
                                         ('native', 'b'), ('stabilized', 'a')]]
        rows += [dict(task='medical_vqa', reference={'variant': 'native', 'category': 'a'},
                      teacher_available=False, reasoning='not validated'),
                 dict(task='camera_view', teacher_available=True)]
        trainer = self.trainer(rows, cot=True)
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 3})
        self.assertEqual(trainer.health_subset_indices, (0, 2, 3))
        random.seed(1234)
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 3})
        self.assertEqual(trainer.health_subset_indices, (0, 2, 3))
        trainer.state.global_step = 1
        trainer.run_periodic_health()
        self.assertEqual(trainer.logs[-1]['health_teacher_loss_expected'], 5)
        self.assertEqual(trainer.logs[-1]['health_teacher_loss_covered'], 4)
        self.assertEqual(trainer.logs[-1]['health_examples'], 3)

    def test_empty_coverage_and_nonfinite_loss_warn_without_stopping(self):
        trainer = self.trainer([dict(task='medical_vqa', teacher_available=False)], cot=True)
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 2})
        trainer.state.global_step = 1
        with self.assertLogs('scripts.train.sft.health', level='WARNING'):
            trainer.run_periodic_health()
        self.assertEqual(trainer.logs[-1]['health_unavailable'], 1)
        self.assertNotIn('health_loss', trainer.logs[-1])
        self.assertFalse(trainer.control.should_training_stop)
        trainer = self.trainer()
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 1})
        trainer._teacher_loss_loop = lambda *a, **k: SimpleNamespace(metrics={'health_loss': math.nan})
        trainer.state.global_step = 1
        with self.assertLogs('scripts.train.sft.health', level='WARNING'):
            trainer.run_periodic_health()
        self.assertEqual(trainer.logs[-1]['health_invalid'], 1)
        self.assertFalse(trainer.control.should_training_stop)

    def test_sustained_value_loss_increase_alerts_and_recovers(self):
        trainer = self.trainer()
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 1})
        for step, value in enumerate((3., 6., 4., 2.), 1):
            trainer.state.global_step = step
            trainer._teacher_loss_loop = lambda *a, **k: SimpleNamespace(metrics={
                'health_loss': 2., 'health_medical_value_loss': value,
                'health_medical_value_tokens': 1, 'health_medical_value_accuracy': .5})
            if step == 3:
                with self.assertLogs('scripts.train.sft.health', level='WARNING'):
                    trainer.run_periodic_health()
            else:
                trainer.run_periodic_health()
        self.assertEqual([r['health_alert'] for r in trainer.logs], [0, 0, 1, 0])
        self.assertEqual(trainer.logs[1]['health_medical_value_loss_baseline_ratio'], 2.)
        self.assertFalse(trainer.control.should_training_stop)

    def test_rng_modes_control_and_metric_cache_restored_on_failure(self):
        trainer = self.trainer()
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 1})
        trainer.state.global_step = 1
        initial = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        control = trainer.control
        metrics = trainer._metrics
        loaders = trainer._eval_dataloaders
        def fail(*args, **kwargs):
            random.random(); np.random.rand(); torch.rand(2)
            trainer.model.eval()
            trainer.control.should_training_stop = True
            trainer._metrics['eval']['entropy'] = [123.]
            raise RuntimeError('backend failed')
        trainer._teacher_loss_loop = fail
        with self.assertRaisesRegex(RuntimeError, 'backend failed'):
            trainer.run_periodic_health()
        self.assertEqual(random.getstate(), initial[0])
        np.testing.assert_array_equal(np.random.get_state()[1], initial[1][1])
        torch.testing.assert_close(torch.get_rng_state(), initial[2])
        self.assertTrue(trainer.model.training)
        self.assertFalse(trainer.model[1].training)
        self.assertIs(trainer.control, control)
        self.assertFalse(control.should_training_stop)
        self.assertIs(trainer._metrics, metrics)
        self.assertEqual(metrics['eval']['entropy'], [3.])
        self.assertIs(trainer._eval_dataloaders, loaders)

    def test_logging_uses_real_optimizer_and_keeps_epoch_loader_cache(self):
        trainer = self.trainer()
        trainer.configure_health({'every_n_steps': 1, 'max_examples': 1})
        optimizer = trainer.optimizer
        handler = trainer.callback_handler
        def loader(dataset):
            self.assertNotIn('eval', trainer._eval_dataloaders)
            trainer._eval_dataloaders['health'] = dataset
            return dataset
        def log(metrics):
            self.assertIs(trainer.optimizer, optimizer)
            self.assertIs(trainer.callback_handler, handler)
            trainer.logs.append(dict(metrics))
        trainer.get_eval_dataloader = loader
        trainer.log = log
        trainer.state.global_step = 1
        trainer.run_periodic_health()
        self.assertEqual(trainer._eval_dataloaders, {'eval': 'full validation loader'})
        self.assertEqual(trainer.callback_handler.eval_dataloader, 'epoch loader')

    def test_invalid_diagnostic_ranges_are_visible_without_stopping(self):
        for values in ({'health_loss': -1.}, {'health_medical_value_accuracy': 1.5},
                       {'health_medical_value_tokens': 0}):
            trainer = self.trainer()
            trainer.configure_health({'every_n_steps': 1, 'max_examples': 1})
            metrics = {'health_loss': 2., 'health_medical_value_loss': 3.,
                       'health_medical_value_accuracy': .5, 'health_medical_value_tokens': 1}
            metrics.update(values)
            trainer._teacher_loss_loop = lambda *a, **k: SimpleNamespace(metrics=metrics)
            trainer.state.global_step = 1
            with self.assertLogs('scripts.train.sft.health', level='WARNING'):
                trainer.run_periodic_health()
            self.assertEqual(trainer.logs[-1]['health_invalid'], 1)
            self.assertFalse(trainer.control.should_training_stop)

    def test_real_teacher_loss_does_not_run_generation_or_evaluate_callbacks(self):
        from datasets import Dataset
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from trl import SFTConfig
        from scripts.train.sft.trainer import MedicalSFTTrainer
        mixin = self.mixin()
        cls = MedicalSFTTrainer if issubclass(MedicalSFTTrainer, mixin) else type('HealthTrainer', (mixin, MedicalSFTTrainer), {})
        model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_positions=8, n_embd=8,
            n_layer=1, n_head=1, bos_token_id=1, eos_token_id=1,
            resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        rows = Dataset.from_list([{'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, 3, 4],
            'reference': {'variant': 'native', 'category': 'injury'}}] * 3)
        def collate(examples):
            return dict(input_ids=torch.tensor([e['input_ids'] for e in examples]),
                        labels=torch.tensor([e['labels'] for e in examples]),
                        attention_mask=torch.ones(len(examples), 4, dtype=torch.long),
                        medical_token_groups=torch.tensor([[0, 0, 1, 2]] * len(examples)))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {str(i): i for i in range(16)}, unk_token='0')), pad_token='0', eos_token='1')
        class Guard(TrainerCallback):
            def on_evaluate(self, *args, **kwargs):
                raise AssertionError('Healthcheck entered checkpoint-selection callbacks')
        with tempfile.TemporaryDirectory() as directory:
            args = SFTConfig(output_dir=directory, use_cpu=True, bf16=False, report_to='none',
                disable_tqdm=True, per_device_eval_batch_size=1,
                dataset_kwargs={'skip_prepare_dataset': True}, remove_unused_columns=False,
                label_names=['labels'])
            with patch('scripts.train.sft.trainer.SFTWeightSync'):
                trainer = cls(model=model, args=args, train_dataset=rows, eval_dataset=rows,
                    data_collator=collate, processing_class=tokenizer, callbacks=[Guard()],
                    medical_evaluator=Mock(side_effect=AssertionError('Generation called')),
                    vllm_server_url='http://unused')
            trainer.configure_health({'every_n_steps': 1, 'max_examples': 2})
            trainer.state.global_step = 1
            trainer._metrics['train']['sentinel'] = [12.]
            model.eval()
            with torch.no_grad():
                batch = collate([rows[0]])
                batch.pop('medical_token_groups')
                result = model(**batch)
                expected_loss = result.loss.item()
                expected_value = torch.nn.functional.cross_entropy(result.logits[:, 1], torch.tensor([3])).item()
            model.train()
            before = torch.get_rng_state().clone()
            trainer.run_periodic_health()
            report = trainer.state.log_history[-1]
            self.assertAlmostEqual(report['health_loss'], expected_loss, places=5)
            self.assertAlmostEqual(report['health_medical_value_loss'], expected_value, places=5)
            self.assertEqual(report['health_medical_value_tokens'], 2)
            self.assertNotIn('eval_loss', report)
            self.assertNotIn('sentinel', report)
            self.assertEqual(trainer._metrics['train']['sentinel'], [12.])
            self.assertFalse(trainer._metrics['eval'])
            self.assertIsNone(trainer.state.best_metric)
            self.assertTrue(model.training)
            torch.testing.assert_close(torch.get_rng_state(), before)


if __name__ == '__main__':
    unittest.main()
