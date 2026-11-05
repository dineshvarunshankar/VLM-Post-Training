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
    from scripts.train.shared.contracts import validate_sft_reliability

    validate_sft_reliability(train_args)

    if train_args.get("fsdp") and train_args.get("fsdp_config", {}).get("activation_checkpointing"):
        from scripts.train.shared.checkpoint_rng import install
        install()

    train_args = {"report_to": "none", **train_args}
    train_args.pop('recovery', None)
    train_args.pop('health_check', None)
    train_args.pop('continuation', None)
    if train_args.pop('early_stopping', None) is not None:
        train_args['restore_callback_states_from_checkpoint'] = True
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


def build_callbacks(train_args):
    from scripts.train.shared.contracts import validate_early_stopping
    validate_early_stopping(train_args)
    policy = train_args.get('early_stopping')
    if policy is None:
        return []
    from transformers import EarlyStoppingCallback
    return [EarlyStoppingCallback(early_stopping_patience=policy['patience'],
                                  early_stopping_threshold=policy.get('threshold', 0.0))]


def save_training_outputs(trainer, processor, output, result, test_dataset=None):
    from scripts.train.shared.runtime import require_training_success
    require_training_success(trainer)
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


def _prepare_sft_data(args, config):
    medical_splits = load_splits(
        args.media_root, args.split_dir, args.catalog, args.prompts,
        sources=config.get('dataset', {}).get('sources'),
        evaluation_sources=config.get('evaluation', {}).get('sources'),
    )
    splits = select_medical_examples(medical_splits, config)
    subject_splits, subject_report = None, None
    if config.get('subject_check', {}).get('enabled'):
        from scripts.train.shared.subject_check import load_subject_check_splits
        subject_splits, subject_report = load_subject_check_splits(
            args.media_root, args.split_dir, config, medical_splits, args.catalog, args.prompts)
    sampling_plan, mixture_report = None, None
    if config.get("answer_format") == "cot":
        from scripts.train.shared.cot_data import build_cot_splits, pin_split_membership
        splits, report = build_cot_splits(splits, media_root=args.media_root,
            catalog=read_yaml(args.catalog), config=config)
        pin_split_membership(report,args.split_dir)
        if subject_splits is not None:
            from scripts.train.shared.subject_check_evaluation import attach_subject_check, pin_subject_check_manifest
            splits, sampling_plan, report = attach_subject_check(splits, report['sampling_plan'], report, subject_splits, subject_report)
            pin_subject_check_manifest(report, args.media_root, args.split_dir)
            return splits, sampling_plan, report
        return splits, report["sampling_plan"], report
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
    if subject_splits is not None:
        from scripts.train.shared.subject_check_evaluation import attach_subject_check
        return attach_subject_check(splits, sampling_plan, mixture_report, subject_splits, subject_report)
    return splits, sampling_plan, mixture_report


def prepare_sft_data(args, config):
    """Resolve the actual launcher data route; stream/hash CoT sources on rank zero."""
    if config.get("answer_format") != "cot" or int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return _prepare_sft_data(args, config)
    from accelerate import PartialState
    from accelerate.utils import broadcast_object_list
    from scripts.train.shared.distributed_local_work import run_local_work
    state = PartialState()
    payload = [None]
    def prepare():
        if not state.is_main_process:
            return
        try:
            payload[0] = {"data": _prepare_sft_data(args, config)}
        except Exception as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    run_local_work('CoT data preparation', prepare)
    broadcast_object_list(payload)
    if "error" in payload[0]:
        raise ValueError("CoT rank-zero data validation failed: " + payload[0]["error"])
    return payload[0]["data"]


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
    callbacks = build_callbacks(train_args)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        train_args["max_steps"] = args.max_steps
    model_config = load_model_config(args)
    splits, sampling_plan, mixture_report = prepare_sft_data(args, config)
    from scripts.train.shared.sft_dataset import build_sft_dataset
    # Check the real Arrow boundary during inspection and before loading weights.
    datasets = {name: build_sft_dataset(rows) for name, rows in splits.items()}
    print(
        json.dumps(
            {"model": model_config, "examples": {k: len(v) for k, v in splits.items()}},
            indent=2,
        )
    )
    if config.get('subject_check', {}).get('enabled'):
        print(json.dumps({'subject_check': mixture_report['subject_check'],
                          'image_processor_kwargs': config['subject_check'].get('image_processor_kwargs', {}),
                          'image_budget_policy': 'native processor defaults unless explicit image kwargs are configured'}, indent=2))
    if args.inspect:
        if config.get("answer_format") == "cot":
            print(json.dumps({"coverage":mixture_report,"example":splits["train"][0],
                              "rendering":"Use python -m scripts.train.shared.cot_inspect with explicit run, split, and processor metadata paths for rendered inputs/targets."},indent=2))
        else:
            print(json.dumps(splits["train"][0], indent=2))
        return 0
    if config.get("answer_format") == "cot":
        from scripts.train.shared.cot_data import require_complete_training, verify_release_manifest
        require_complete_training(mixture_report)
        if not args.check_batch:
            verify_release_manifest(config, mixture_report)

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
    from scripts.train.shared.config import evaluation_request_timeout
    from scripts.train.shared.eval_replicas import replica_plan_from_environment, replica_for_rank
    replica_plan = replica_plan_from_environment(int(os.environ.get('WORLD_SIZE', '1')))
    if replica_plan is not None and not args.vllm_gpu_sharing:
        raise ValueError('Replicated evaluation requires shared GPU lifecycle')
    endpoints = ([item['url'] for item in replica_plan['replicas']]
                 if replica_plan is not None else [args.vllm_server_url])
    clients = [VideoRolloutClient(url, args.vllm_model_name,
                                 api_key=os.environ.get('VLLM_API_KEY'),
                                 timeout=evaluation_request_timeout(config)) for url in endpoints]
    def check_endpoints():
        for endpoint in clients:
            endpoint.check_compatibility()
            if config.get('subject_check', {}).get('enabled'):
                endpoint.check_image_compatibility()
            if config.get('answer_format') == 'cot' or config.get('subject_check', {}).get('enabled'):
                from scripts.train.shared.cot_budget import PARSERS
                from scripts.train.shared.cot_format import model_family
                endpoint.check_cot_schema_compatibility(PARSERS[model_family(checkpoint_config)])
    if replica_plan is not None:
        from accelerate import PartialState
        from scripts.train.shared.distributed_local_work import run_local_work
        runtime_state = PartialState()
        run_local_work('replica compatibility', check_endpoints)
        client = clients[replica_for_rank(replica_plan, runtime_state.process_index)['replica_id']]
    else:
        check_endpoints()
        client = clients[0]
    args.continuation_policy = train_args.get("continuation")
    if args.continuation_policy is not None:
        if not args.resume:
            raise ValueError("Continuation requires an explicit resume checkpoint")
        from scripts.train.sft.continuation import validate_checkpoint, validate_completion_outputs
        validate_checkpoint(args.resume, fsdp=bool(train_args.get("fsdp")),
                            world_size=int(os.environ.get("WORLD_SIZE", "1")),
                            require_health=bool(train_args.get("health_check")),
                            require_weights=True, policy=args.continuation_policy)
        validate_completion_outputs(args.resume.resolve().parent.parent)
    output = validate_runtime(args, model_config, answer_format=config.get("answer_format","answer_only"))
    evaluation_lifecycle = None
    if args.vllm_gpu_sharing:
        from accelerate import PartialState
        from scripts.train.shared.eval_lifecycle import SharedEvaluationServer
        evaluation_lifecycle = SharedEvaluationServer(clients if replica_plan is not None else client, PartialState())
        # Distributed state is ready; no model weights have been loaded yet.
        evaluation_lifecycle.initialize()
    from scripts.train.shared.tracking import configure_tracking, add_tracking_metadata
    train_args.update(configure_tracking(args, "sft_only", output, repository=model_config["repository"]))
    model, adapter_summary = load_lora_model(model_config, checkpoint_config, config)
    model.enable_input_require_grads()
    training = build_training_config(output, train_args)
    from scripts.train.shared.config import evaluation_settings

    evaluation = evaluation_settings(config, model_config)
    evaluator_class = MedicalVLLMEvaluator
    evaluator_options = {'subject_check_rows': (mixture_report or {}).get('subject_check_evaluation', {})}
    if config.get('subject_check', {}).get('enabled'):
        from scripts.train.shared.cot_budget import PARSERS
        from scripts.train.shared.cot_format import model_family
        evaluator_options['subject_reasoning_parser'] = PARSERS[model_family(checkpoint_config)]
    if config.get("answer_format") == "cot":
        from scripts.train.shared.cot_evaluation import CoTVLLMEvaluator
        from scripts.train.shared.cot_format import model_family
        evaluator_class = CoTVLLMEvaluator
        evaluator_options.update(model_family=model_family(checkpoint_config),coverage_report=mixture_report)
    evaluator = evaluator_class(
        client, feature_configuration(checkpoint_config, processor), processor,
        read_yaml(args.catalog), evaluation["generation"],
        f"{args.run_date}:{args.model}:{args.run_name}",
        max_tokens=evaluation["max_new_tokens"],
        experiment=config, medical_templates=read_yaml(args.prompts),
        lifecycle=evaluation_lifecycle,
        **evaluator_options,
    )
    if replica_plan is not None:
        evaluator.request_batch_size = replica_plan["request_batch_size"]
    trainer = MedicalSFTTrainer(
        callbacks=callbacks,
        model=model,
        args=training,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        processing_class=processor,
        medical_evaluator=evaluator,
        vllm_server_url=args.vllm_server_url,
        train_sampling_plan=sampling_plan,
        mixture_config=config.get("mixture"),
        recovery_policy=train_args.get("recovery"),
        health_policy=train_args.get("health_check"),
        continuation_policy=train_args.get("continuation"),
    )
    if trainer.is_world_process_zero():
        trainer.continuation_metadata = save_run_metadata(
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
    from scripts.train.sft.health import HealthDeteriorationStop
    try:
        result = trainer.train(
            resume_from_checkpoint=str(args.resume) if args.resume else None
        )
    except HealthDeteriorationStop:
        from scripts.train.shared.runtime import finish_deteriorated_training
        finish_deteriorated_training(trainer)
        return 2
    save_training_outputs(
        trainer, processor, output, result,
        datasets["test"] if splits["test"] else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
