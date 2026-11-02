import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from scripts.train.shared.evaluation import MedicalVLLMEvaluator
from scripts.train.tests import test_shared_eval_lifecycle as lifecycle_tests


class SharedEvaluationIntegrationTests(unittest.TestCase):
    setUp = lifecycle_tests.SharedEvalTests.setUp
    tearDown = lifecycle_tests.SharedEvalTests.tearDown

    def test_real_lifecycle_wraps_metrics_and_returns_server_to_sleep(self):
        evaluator = MedicalVLLMEvaluator(self.client, {}, None,
            {'categories': {'medical': {}}}, {}, 'test-run', lifecycle=self.lifecycle)
        evaluator.auxiliary = SimpleNamespace(prepare=lambda dataset, prefix: ([], {}), scoring={})
        def predict(trainer, rows, *args):
            self.assertFalse(self.sleeping)
            self.events.append('predict')
            return ['{}'] * len(rows)
        evaluator.predict_many = predict
        rows = [{'reference': {'category': 'medical'}, 'answer': '{}'}]
        trainer = SimpleNamespace(
            vllm_generation=SimpleNamespace(sync_weights=lambda: self.events.append('sync')),
            state=SimpleNamespace(global_step=1), args=SimpleNamespace(seed=42),
            accelerator=SimpleNamespace(process_index=0, num_processes=1,
                                        wait_for_everyone=lambda: None),
            is_world_process_zero=lambda: False, log=Mock(), control=object(),
            callback_handler=SimpleNamespace(on_evaluate=Mock(return_value='updated')))
        self.lifecycle.initialize()
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
        self.assertEqual(trainer.control, 'updated')
        self.assertTrue(self.sleeping)
        self.assertLess(self.events.index('sync'), self.events.index('predict'))
        self.assertEqual(self.events[-1], '/sleep?level=1')

if __name__ == '__main__':
    unittest.main()
