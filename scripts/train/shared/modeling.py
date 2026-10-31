"""Shared model loading and video processor construction."""

from pathlib import Path

from scripts.train.shared.data import read_yaml

def load_model_config(args):
    model_config = dict(read_yaml(args.models)["models"][args.model])
    if args.checkpoint:
        model_config["local_path"] = str(Path(args.checkpoint).expanduser().resolve())
    if args.revision:
        model_config["revision"] = args.revision
    if "generation" not in model_config:
        raise ValueError("Model profile needs explicit generation settings")
    if not args.inspect:
        from scripts.train.storage.cloud_models import verify_local_source

        if "local_path" not in model_config:
            if args.weights_root is None:
                raise ValueError("Set LOCAL_WEIGHTS_ROOT or pass --weights-root")
            model_config["local_path"] = str(
                (
                    args.weights_root.expanduser()
                    / model_config["repository"]
                    / model_config["revision"]
                ).resolve()
            )
        source = verify_local_source(
            Path(model_config["local_path"]), model_config, check_hashes=False
        )
        model_config["cloud_source"] = source["source"]
    return model_config


def build_processor(model_config, config):
    from scripts.train.shared.collator import AnswerOnlyCollator
    from transformers import AutoConfig, AutoProcessor
    from scripts.train.shared.video import spatial_options

    location = model_config["local_path"]
    options = {"local_files_only": True, "trust_remote_code": False}
    checkpoint_config = AutoConfig.from_pretrained(location, **options)
    text_config = getattr(checkpoint_config, "text_config", checkpoint_config)
    context_limit = getattr(text_config, "max_position_embeddings", None)
    if type(config["max_length"]) is not int or config["max_length"] <= 0:
        raise ValueError("max_length must be a positive integer")
    if context_limit is None or config["max_length"] > context_limit:
        raise ValueError(
            f"max_length={config['max_length']} exceeds or cannot be checked against checkpoint capacity {context_limit}"
        )
    processor = AutoProcessor.from_pretrained(location, **options)
    collator = AnswerOnlyCollator(
        processor,
        config["video"]["fps"],
        config["max_length"],
        spatial_options(processor, config["video"]),
    )
    return checkpoint_config, processor, collator


def load_lora_model(model_config, checkpoint_config, config):
    import torch
    import transformers
    from scripts.train.shared.adapters import select_lora_targets, verify_adapter_parameters
    from peft import LoraConfig, get_peft_model
    from scripts.train.shared.peft_compat import install_checkpoint_adapter_compatibility
    from scripts.train.shared.video_compat import install_video_frame_compatibility

    install_checkpoint_adapter_compatibility()

    location = model_config["local_path"]
    options = {"local_files_only": True, "trust_remote_code": False}
    architecture = checkpoint_config.architectures
    if (
        not architecture
        or len(architecture) != 1
        or not hasattr(transformers, architecture[0])
    ):
        raise ValueError(
            "Checkpoint architecture is not available in the locked Transformers version"
        )
    model_class = getattr(transformers, architecture[0])
    model = model_class.from_pretrained(
        location,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attention", "sdpa"),
        **options,
    )
    install_video_frame_compatibility(model)
    # Gemma4 declares an optional audio block even for checkpoints without audio.
    # Accelerate resolves every declared class before FSDP wrapping. Keep this
    # correction instance-local and preserve errors for other missing classes.
    if (model.config.model_type == 'gemma4'
            and model.config.audio_config is None
            and not any(type(module).__name__ == 'Gemma4AudioLayer' for module in model.modules())):
        model._no_split_modules = [
            name for name in model._no_split_modules if name != 'Gemma4AudioLayer'
        ]
    model.config.use_cache = False
    targets, groups = select_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(task_type="CAUSAL_LM", target_modules=targets, **config["lora"]),
        autocast_adapter_dtype=True,
    )
    summary = {
        "target_modules": targets,
        "components": groups,
        "trainable_parameters": verify_adapter_parameters(model),
    }
    model.print_trainable_parameters()
    return model, summary
