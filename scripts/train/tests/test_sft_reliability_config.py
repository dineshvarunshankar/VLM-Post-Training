"""Operational safety policies remain separate from optimization and model selection."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts.train.shared import contracts
from scripts.train.sft.launch import build_training_config


class ReliabilityConfigTests(unittest.TestCase):
    def policies(self):
        return {'recovery': {'before_evaluation': True, 'save_steps': 100, 'save_total_limit': 2},
                'health_check': {'every_n_steps': 100, 'max_examples': 32}}

    def test_validation_rejects_invalid_or_unknown_safety_settings(self):
        self.assertTrue(hasattr(contracts, 'validate_sft_reliability'), 'Safety policies are not validated')
        contracts.validate_sft_reliability({})
        contracts.validate_sft_reliability(self.policies())
        bad = [
            {'recovery': {'before_evaluation': True, 'save_steps': 0, 'save_total_limit': 2}},
            {'recovery': {'before_evaluation': 'yes', 'save_steps': 100, 'save_total_limit': 2}},
            {'recovery': {'before_evaluation': True, 'save_steps': 100, 'save_total_limit': True}},
            {'health_check': {'every_n_steps': -1, 'max_examples': 32}},
            {'health_check': {'every_n_steps': 100, 'max_examples': False}},
            {'health_check': {'every_n_steps': 100, 'max_examples': 32, 'stop_training': True}},
        ]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                contracts.validate_sft_reliability(value)

    def test_extensions_do_not_reach_hf_or_change_best_checkpoint_policy(self):
        values = dict(use_cpu=True, eval_strategy='epoch', save_strategy='epoch',
                      load_best_model_at_end=True, metric_for_best_model='macro_f1',
                      greater_is_better=True, learning_rate=2e-5, **self.policies())
        original = copy.deepcopy(values)
        with tempfile.TemporaryDirectory() as output:
            args = build_training_config(Path(output), values)
        self.assertEqual(values, original)
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.eval_strategy, 'epoch')
        self.assertEqual(args.save_strategy, 'epoch')
        self.assertEqual(args.metric_for_best_model, 'macro_f1')
        self.assertTrue(args.load_best_model_at_end)
        self.assertFalse(hasattr(args, 'health_check'))
        self.assertFalse(hasattr(args, 'recovery'))

class RecoveryResumeValidationTests(unittest.TestCase):
    def test_partial_recovery_is_rejected_and_completed_recovery_is_accepted(self):
        import json
        from scripts.train.shared.runtime import validate_output_directory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'recovery/checkpoint-100'
            checkpoint.mkdir(parents=True)
            (checkpoint/'trainer_state.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'recovery'):
                validate_output_directory(root, checkpoint)
            marker = checkpoint/'recovery_complete.json'
            for payload in ({'schema_version':1,'global_step':99,'completed':True},
                            {'schema_version':1,'global_step':100,'completed':False}):
                marker.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, 'recovery'):
                    validate_output_directory(root, checkpoint)
            marker.write_text(json.dumps({'schema_version':1,'global_step':100,'completed':True}))
            validate_output_directory(root, checkpoint)

    def test_normal_checkpoint_does_not_require_recovery_marker(self):
        from scripts.train.shared.runtime import validate_output_directory
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            checkpoint=root/'checkpoints/checkpoint-100'
            checkpoint.mkdir(parents=True)
            (checkpoint/'trainer_state.json').write_text('{}')
            validate_output_directory(root,checkpoint)


if __name__ == '__main__':
    unittest.main()
