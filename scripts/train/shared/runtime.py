"""Shared run validation, resume checks, and metadata persistence."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from scripts.train.shared.data import atomic_json, sha256
from scripts.train.shared.config import evaluation_settings


def validate_recovery_checkpoint(resume):
    """Require atomic completion evidence only for explicit recovery resumes."""
    checkpoint = Path(resume).resolve()
    if (checkpoint.parent.parent.name == 'recovery'
            and checkpoint.parent.name.startswith('.incomplete-')):
        raise ValueError('Cannot resume an incomplete recovery checkpoint')
    if checkpoint.parent.name != 'recovery':
        return
    try:
        marker = json.loads((checkpoint / 'recovery_complete.json').read_text())
        step = int(checkpoint.name.removeprefix('checkpoint-'))
        if (not checkpoint.name.startswith('checkpoint-')
                or not isinstance(marker, dict)
                or type(marker.get('schema_version')) is not int
                or marker['schema_version'] != 1
                or marker.get('completed') is not True
                or type(marker.get('global_step')) is not int
                or marker['global_step'] != step):
            raise ValueError('Invalid completion marker')
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f'Cannot resume incomplete or invalid recovery checkpoint: {checkpoint}') from exc


def validate_output_directory(output, resume):
    """Share one output decision before any rank can create run artifacts."""
    import torch.distributed as distributed

    collective = distributed.is_available() and distributed.is_initialized()
    error = None
    if not collective or distributed.get_rank() == 0:
        try:
            if output.exists() and any(output.iterdir()) and not resume:
                error = "Output directory is nonempty; choose a new directory or use --resume"
            elif resume and not (resume / "trainer_state.json").is_file():
                error = "--resume must point to a Trainer checkpoint containing trainer_state.json"
            elif resume:
                try:
                    validate_recovery_checkpoint(resume)
                except ValueError as exc:
                    error = str(exc)
        except OSError as exc:
            error = f"Cannot validate output directory: {exc}"
    if collective:
        result = [error]
        distributed.broadcast_object_list(result, src=0)
        error = result[0]
        distributed.barrier()
    if error:
        raise ValueError(error)


def validate_runtime(args, model_config, recipe="sft_only", *, answer_format="answer_only"):
    import torch
    from scripts.train.storage.cloud_models import run_destination

    if args.resume:
        previous_run = args.resume.resolve().parent.parent / "run.json"
        previous = json.loads(previous_run.read_text())
        args.run_location = previous["run_location"]
        if args.run_location.get("recipe", recipe) != recipe or args.run_location.get("answer_format", "answer_only") != answer_format:
            raise ValueError("Resume must use the original recipe and answer format")
        args.run_name = previous["run_location"]["name"]
        args.run_date = previous["run_location"]["date"]
        if args.output_dir is not None and args.output_dir.resolve() != previous_run.parent:
            raise ValueError('Resume must use the original run directory; choose a new run to change its configuration')
        args.output_dir = previous_run.parent
    if not args.run_name:
        raise ValueError("--run-name is required for training")
    if not args.run_date:
        raise ValueError("--run-date YYYY-MM-DD is required for training")
    if not args.resume:
        args.run_location = {"date": args.run_date, "name": args.run_name,
                             "answer_format": answer_format, "recipe": recipe}
    relative = run_destination(
        "runs", model_config["repository"], args.run_date, args.run_name,
        answer_format=args.run_location.get("answer_format"), recipe=args.run_location.get("recipe"),
    )
    if args.output_dir is None:
        if args.output_root is None:
            raise ValueError("Set LOCAL_WEIGHTS_ROOT or pass --output-root")
        args.output_dir = args.output_root.expanduser().joinpath(
            *Path(relative).parts[1:]
        )
    output = args.output_dir.resolve()
    if not torch.cuda.is_available():
        raise ValueError(
            "Training requires CUDA; use --inspect or --check-batch for CPU checks"
        )
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        raise ValueError(
            "Use torchrun for multiple GPUs, or expose one GPU with CUDA_VISIBLE_DEVICES, to preserve the effective batch size"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This configuration requires BF16-capable GPUs")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        from accelerate import PartialState
        PartialState()
    validate_output_directory(output, args.resume)
    return output


def save_run_metadata(
    args,
    output,
    model_config,
    config,
    train_args,
    adapter_summary,
    splits,
    mixture_report=None,
):
    import accelerate
    import peft
    import torch
    import transformers
    import trl

    output.mkdir(parents=True, exist_ok=True)
    run = {
        "run_location": args.run_location,
        "model": model_config,
        "config": config,
        "training_arguments": train_args,
        "evaluation_backend": "vllm",
        "adapters": adapter_summary,
        "data_sha256": {
            name: hashlib.sha256(
                json.dumps(
                    [{k: v for k, v in row.items() if k != "video"} for row in rows],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for name, rows in splits.items()
        },
        "catalog_sha256": sha256(args.catalog),
        "prompts_sha256": sha256(args.prompts),
        "split_sha256": {
            name: sha256(args.split_dir / (name + ".jsonl")) for name in splits
        },
        "libraries": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
        },
    }
    launch_config = os.environ.get('TRIAGE_RUN_CONFIG')
    if launch_config:
        from scripts.train.shared.data import read_yaml
        from scripts.train.shared.run_config import reject_secrets
        run['launch_config'] = read_yaml(launch_config)
        reject_secrets(run['launch_config'])
    if mixture_report is not None:
        run["mixture"] = mixture_report
    if config.get("answer_format")=="cot":
        from scripts.train.shared.cot_budget import generation_policy, MODEL_PARSERS
        run["generation_budget_policy"] = generation_policy(config,reasoning_parser=MODEL_PARSERS[args.model])
    if args.resume:
        previous_path = args.resume.resolve().parent.parent / "run.json"
        previous = json.loads(previous_path.read_text())
        if previous != run:
            raise ValueError(
                "Resume configuration, prompts, or split files differ from run.json"
            )
    if config.get("answer_format")=="cot":
        atomic_json(output / "cot_generation_policy.json",run["generation_budget_policy"])
    atomic_json(output / "run.json", run)
    atomic_json(output / "generation_policy.json", evaluation_settings(config, model_config)["generation"])
    shutil.copy2(args.catalog, output / "label_catalog.yaml")
    shutil.copy2(args.prompts, output / "prompts.yaml")
    if mixture_report is not None:
        from scripts.train.shared.auxiliary import PROMPT_DIRECTORY, PROMPTS

        destination = output / "task_prompts"
        destination.mkdir(exist_ok=True)
        for name in PROMPTS:
            shutil.copy2(
                PROMPT_DIRECTORY / (name + ".yaml"), destination / (name + ".yaml")
            )



def finish_training_and_export(trainer, output):
    """Finish distributed work before rank zero performs the standalone CPU export."""
    is_main = trainer.is_world_process_zero()
    trainer.accelerator.wait_for_everyone()
    from scripts.train.shared.tracking import finish_tracking
    finish_tracking(trainer)
    trainer.accelerator.end_training()
    if is_main:
        from scripts.train.storage.export_model import export_saved_run

        export_saved_run(output)


def evaluate_heldout(trainer, dataset, output):
    """Score the restored best checkpoint without using test scores for selection."""
    if dataset is None or len(dataset) == 0:
        report = {'status': 'skipped', 'reason': 'No eligible held-out test examples'}
    else:
        checkpoint = trainer.state.best_model_checkpoint
        validation_score = trainer.state.best_metric
        if not trainer.args.load_best_model_at_end or not checkpoint or validation_score is None:
            raise ValueError('Test evaluation requires a restored validation-selected best checkpoint')
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix='test')
        if (trainer.state.best_model_checkpoint != checkpoint
                or trainer.state.best_metric != validation_score):
            raise RuntimeError('Test evaluation changed validation checkpoint selection')
        report = {
            'status': 'completed', 'checkpoint': checkpoint,
            'validation_score': validation_score, 'examples': len(dataset),
            'metrics': metrics,
            'prediction_report': f'checkpoints/test-step-{trainer.state.global_step}.json',
        }
    if trainer.is_world_process_zero():
        atomic_json(Path(output) / 'test_results.json', report)
    return report
