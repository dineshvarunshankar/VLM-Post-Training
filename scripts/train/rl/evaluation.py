"""Baseline-inclusive checkpoint selection for RL."""

from pathlib import Path
from transformers import TrainerCallback


class BaselineCheckpoint(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step == 0 and state.best_metric is None:
            metrics = self.trainer.evaluate()
            state.best_metric = metrics['eval_macro_f1']
            state.best_global_step = 0
            state.best_model_checkpoint = str(Path(args.output_dir) / 'checkpoint-0')
            self.trainer._save_checkpoint(self.trainer.model, trial=None)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control
