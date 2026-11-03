"""Optional stopping follows medical F1 and keeps callback state on resume."""
from pathlib import Path
from types import SimpleNamespace
import unittest

from transformers import TrainerControl, TrainerState
from scripts.train.sft import launch


class EarlyStoppingTests(unittest.TestCase):
    def test_patience_uses_medical_f1_not_loss_and_preserves_resume_setting(self):
        self.assertTrue(hasattr(launch, 'build_callbacks'), 'Early stopping is not wired')
        values = dict(use_cpu=True, eval_strategy='epoch', save_strategy='epoch',
                      load_best_model_at_end=True, metric_for_best_model='macro_f1',
                      greater_is_better=True, early_stopping={'patience': 2, 'threshold': 0.0})
        args = launch.build_training_config(Path('/tmp/early-stopping-test'), values)
        self.assertTrue(args.restore_callback_states_from_checkpoint)
        callback, = launch.build_callbacks(values)
        state = TrainerState(best_metric=.5)
        control = TrainerControl()
        callback.on_evaluate(args, state, control, metrics={'eval_macro_f1': .49, 'eval_loss': .001})
        self.assertFalse(control.should_training_stop)
        callback.on_evaluate(args, state, control, metrics={'eval_macro_f1': .48, 'eval_loss': .0001})
        self.assertTrue(control.should_training_stop)
        self.assertEqual(launch.build_callbacks({}), [])

    def test_invalid_policy_fails_before_model_loading(self):
        self.assertTrue(hasattr(launch, 'build_callbacks'), 'Early stopping is not wired')
        for policy in ({'patience': 0}, {'patience': True}, {'patience': 2, 'threshold': -1}):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                launch.build_callbacks({'early_stopping': policy})
        with self.assertRaises(ValueError):
            launch.build_callbacks({'early_stopping': {'patience': 2}, 'load_best_model_at_end': False})


if __name__ == '__main__':
    unittest.main()
