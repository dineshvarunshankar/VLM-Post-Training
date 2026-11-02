#!/usr/bin/env python3
"""Answer-only video SFT with TRL and PEFT LoRA."""


import argparse
import json
import os
from scripts.train.shared.storage_settings import local_weights_directory, local_dataset_directory
from pathlib import Path

from scripts.train.shared.paths import REPO_ROOT, CONFIG_DIR
from scripts.train.shared.data import atomic_json, load_splits, read_yaml
from scripts.train.shared.config import (
    load_training_config,
    select_medical_examples,
)


from scripts.train.shared.modeling import load_model_config, build_processor, load_lora_model
from scripts.train.shared.runtime import validate_runtime, save_run_metadata


def resolve_training_args(config, data_parallel_size):
    args = dict(config["training"])
    effective = config["effective_batch_size"]
    per_device = args["per_device_train_batch_size"]
    if any(
        type(value) is not int or value <= 0
        for value in (effective, per_device, data_parallel_size)
    ):
        raise ValueError("Batch sizes and data-parallel size must be positive integers")
    batch_per_step = per_device * data_parallel_size
    if effective % batch_per_step:
        raise ValueError(
            "Effective batch size must be divisible by per-device batch size times data-parallel size"
        )
    args["gradient_accumulation_steps"] = effective // batch_per_step
    if args["dataloader_num_workers"] == 0:
        args.pop("dataloader_prefetch_factor", None)
        args.pop("dataloader_multiprocessing_context", None)
    return args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Resolved SFT stage YAML produced by run.py"
    )
    parser.add_argument("--model", required=True, help="Key in configs/models.yaml")
    parser.add_argument(
        "--models", type=Path, default=CONFIG_DIR / "models.yaml"
    )
    parser.add_argument("--media-root", type=Path, default=local_dataset_directory(), required=local_dataset_directory() is None)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Explicit local run directory")
    parser.add_argument(
        "--output-root", type=Path, default=local_weights_directory("finetuned")
    )
    parser.add_argument("--run-name", help="Name used in the dated cloud output path")
    parser.add_argument(
        "--run-date",
        help="Required run date: YYYY-MM-DD (restored automatically on resume)",
    )
    parser.add_argument(
        "--weights-root", type=Path, default=local_weights_directory("base_weights")
    )
    parser.add_argument("--checkpoint", help="Verified local AirLab model directory")
    parser.add_argument("--vllm-server-url", default=os.environ.get("VLLM_SERVER_URL"))
    parser.add_argument("--vllm-gpu-sharing", action="store_true",
                        help="Sleep the evaluation server while training on its GPUs")
    parser.add_argument("--vllm-model-name", default=os.environ.get("VLLM_MODEL_NAME"))
    parser.add_argument("--revision", help="Revision for an overridden checkpoint")
    parser.add_argument(
        "--resume", type=Path, help="SFT checkpoint directory to resume"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Validate targets/splits and print one text example; no model loading",
    )
    parser.add_argument(
        "--check-batch",
        action="store_true",
        help="Load processor only and check one video/answer batch",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override optimizer steps for a short training smoke test",
    )
    return parser.parse_args()


def check_batch(collator, processor, examples):
    batch = collator(examples[:1])
    supervised = batch["input_ids"][0][batch["labels"][0] != -100]
    print(
        json.dumps(
            {
                "batch_shapes": {
                    k: list(v.shape) for k, v in batch.items() if hasattr(v, "shape")
                },
                "supervised_text": processor.tokenizer.decode(supervised),
            },
            indent=2,
        )
    )


def build_training_config(output, train_args):
    from trl import SFTConfig

    train_args = {"report_to": "none", **train_args}
    train_args["per_device_eval_batch_size"] = 1
    checkpointing = train_args.pop("gradient_checkpointing", not train_args.get("fsdp", False))
    return SFTConfig(
        output_dir=str(output / "checkpoints"),
        remove_unused_columns=False,
        label_names=["labels"],
        bf16=True,
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        max_length=None,  # The video collator enforces the configured limit without truncation.
        **train_args,
    )


def save_training_outputs(trainer, processor, output, result, test_dataset=None):
    trainer.save_model(str(output / "adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output / "adapter")
        atomic_json(output / "train_metrics.json", result.metrics)
        atomic_json(
            output / "checkpoint_selection.json",
            {
                "metric": trainer.args.metric_for_best_model,
                "best_score": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "latest_checkpoint": str(
                    Path(trainer.args.output_dir)
                    / f"checkpoint-{trainer.state.global_step}"
                ),
                "final_adapter": str(output / "adapter"),
            },
        )

    from scripts.train.shared.runtime import evaluate_heldout, finish_training_and_export

    evaluate_heldout(trainer, test_dataset, output)
    finish_training_and_export(trainer, output)


def main():
    from dotenv import load_dotenv

    if not os.environ.get("TRIAGE_RUN_CONFIG"):
        load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    config = load_training_config(args.config)
    from scripts.train.shared.config import validate_grounding_readiness
    validate_grounding_readiness(config)
    if config.get("recipe") != "sft_only":
        raise ValueError("train_sft.py requires an sft_only configuration")
    if config["method"] != "lora":
        raise ValueError("This training script supports LoRA only")
    train_args = resolve_training_args(config, int(os.environ.get("WORLD_SIZE", "1")))
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        train_args["max_steps"] = args.max_steps
    model_config = load_model_config(args)
    splits = select_medical_examples(
        load_splits(
            args.media_root,
            args.split_dir,
            args.catalog,
            args.prompts,
            sources=config.get("dataset", {}).get("sources"),
            evaluation_sources=config.get("evaluation", {}).get("sources"),
        ),
        config,
    )
    sampling_plan, mixture_report = None, None
    if "mixture" in config:
        from scripts.train.shared.auxiliary import build_mixture

        splits["train"], sampling_plan, mixture_report = build_mixture(
            splits["train"], read_yaml(args.catalog), read_yaml(args.prompts), config
        )
        print(
            json.dumps(
                {
                    **{
                        k: v
                        for k, v in mixture_report.items()
                        if k != "excluded_candidates"
                    },
                    "excluded_candidate_count": len(
                        mixture_report["excluded_candidates"]
                    ),
                    "excluded_candidate_preview": mixture_report["excluded_candidates"][
                        :10
                    ],
                },
                indent=2,
            )
        )
    print(
        json.dumps(
            {"model": model_config, "examples": {k: len(v) for k, v in splits.items()}},
            indent=2,
        )
    )
    if args.inspect:
        print(json.dumps(splits["train"][0], indent=2))
        return 0

    from datasets import Dataset
    from scripts.train.sft.trainer import MedicalSFTTrainer
    from transformers import set_seed

    set_seed(config["training"]["seed"])
    checkpoint_config, processor, collator = build_processor(model_config, config)
    if args.check_batch:
        check_batch(collator, processor, splits["train"])
        return 0

    if not args.vllm_server_url or not args.vllm_model_name:
        raise ValueError("SFT evaluation requires VLLM_SERVER_URL and VLLM_MODEL_NAME (or their CLI options)")
    from scripts.train.shared.evaluation import MedicalVLLMEvaluator
    from scripts.train.shared.vllm import feature_configuration
    from scripts.train.rl.rollout.client import VideoRolloutClient
    client = VideoRolloutClient(args.vllm_server_url, args.vllm_model_name,
                               api_key=os.environ.get("VLLM_API_KEY"))
    client.check_compatibility()
    output = validate_runtime(args, model_config)
    evaluation_lifecycle = None
    if args.vllm_gpu_sharing:
        from accelerate import PartialState
        from scripts.train.shared.eval_lifecycle import SharedEvaluationServer
        evaluation_lifecycle = SharedEvaluationServer(client, PartialState())
        # Distributed state is ready; no model weights have been loaded yet.
        evaluation_lifecycle.initialize()
    from scripts.train.shared.tracking import configure_tracking, add_tracking_metadata
    train_args.update(configure_tracking(args, "sft_only", output, repository=model_config["repository"]))
    model, adapter_summary = load_lora_model(model_config, checkpoint_config, config)
    model.enable_input_require_grads()
    training = build_training_config(output, train_args)
    from scripts.train.shared.config import evaluation_settings

    evaluation = evaluation_settings(config, model_config)
    evaluator = MedicalVLLMEvaluator(
        client, feature_configuration(checkpoint_config, processor), processor,
        read_yaml(args.catalog), evaluation["generation"],
        f"{args.run_date}:{args.model}:{args.run_name}",
        max_tokens=evaluation["max_new_tokens"],
        experiment=config, medical_templates=read_yaml(args.prompts),
        lifecycle=evaluation_lifecycle,
    )
    trainer = MedicalSFTTrainer(
        model=model,
        args=training,
        train_dataset=Dataset.from_list(splits["train"]),
        eval_dataset=Dataset.from_list(splits["validation"]),
        data_collator=collator,
        processing_class=processor,
        medical_evaluator=evaluator,
        vllm_server_url=args.vllm_server_url,
        train_sampling_plan=sampling_plan,
        mixture_config=config.get("mixture"),
    )
    if trainer.is_world_process_zero():
        save_run_metadata(
            args,
            output,
            model_config,
            config,
            train_args,
            adapter_summary,
            splits,
            mixture_report,
        )
    add_tracking_metadata(trainer, output)
    result = trainer.train(
        resume_from_checkpoint=str(args.resume) if args.resume else None
    )
    save_training_outputs(
        trainer, processor, output, result,
        Dataset.from_list(splits["test"]) if splits["test"] else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
