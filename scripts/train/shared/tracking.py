"""W&B configuration shared by SFT and both RL recipes."""

import hashlib
import json
import os
from pathlib import Path
from scripts.train.shared.storage_settings import storage_settings
from scripts.train.storage.cloud_models import run_destination


def configure_tracking(args, recipe, output, *, repository=None):
    """Configure Trainer reporting without starting a run during inspection."""
    settings = storage_settings()['wandb']
    mode = os.environ.get('WANDB_MODE', settings['mode']).strip().lower()
    if mode not in {'online', 'offline', 'disabled'}:
        raise ValueError('WANDB_MODE must be online, offline, or disabled')
    name = run_destination('runs', repository or args.model, args.run_date, args.run_name,
                           answer_format='answer_only', recipe=recipe).removeprefix('runs/')
    if args.resume and hasattr(args, 'run_location') and 'answer_format' not in args.run_location:
        name = f'{args.run_date}/{args.model}/{recipe}/{args.run_name}'
    if mode == 'disabled':
        return {'report_to': 'none', 'run_name': name}
    defaults = {'WANDB_' + key.upper(): str(settings[key])
                for key in ('entity', 'project', 'log_model', 'watch')}
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
    os.environ['WANDB_MODE'] = mode
    os.environ['WANDB_RUN_ID'] = hashlib.sha256(name.encode()).hexdigest()[:24]
    os.environ['WANDB_RESUME'] = 'allow' if args.resume else 'never'
    directory = Path(output) / 'tracking'
    directory.mkdir(parents=True, exist_ok=True)
    os.environ['WANDB_DIR'] = str(directory)
    return {'report_to': 'wandb', 'run_name': name}


def add_tracking_metadata(trainer, output):
    """Add recipe and dataset fingerprints to the native Trainer W&B run."""
    if 'wandb' not in trainer.args.report_to:
        return
    from transformers import TrainerCallback

    class RunMetadataCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            import wandb

            metadata = json.loads((Path(output) / 'run.json').read_text())
            wandb.config.update({
                'triage': {key: metadata[key] for key in (
                    'run_location', 'config', 'model', 'adapters', 'libraries',
                    'catalog_sha256', 'prompts_sha256', 'split_sha256', 'data_sha256',
                )}
            }, allow_val_change=True)

    trainer.add_callback(RunMetadataCallback())


def finish_tracking(trainer):
    """Flush metrics after held-out evaluation and before merged-model export."""
    if trainer.is_world_process_zero() and 'wandb' in trainer.args.report_to:
        import wandb

        if wandb.run is not None:
            wandb.summary.update({
                'best_validation_macro_f1': trainer.state.best_metric,
                'best_checkpoint': trainer.state.best_model_checkpoint,
            })
            wandb.finish()
