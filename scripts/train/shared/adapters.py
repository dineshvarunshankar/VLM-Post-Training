"""Explicit LoRA coverage for the selected RGB video model architectures."""

import re
from collections import Counter

import torch

_DECODER = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
_TARGETS = {
    "qwen3_5": {
        "decoder": _DECODER
        + r"|model\.language_model\.layers\.\d+\.linear_attn\.(?:out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)",
        "vision_encoder": r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))",
        "visual_connector": r"model\.visual\.merger\.(?:linear_fc1|linear_fc2)",
    },
    "gemma4_unified": {
        "decoder": _DECODER,
        "visual_connector": r"model\.embed_vision\.(?:patch_dense|multimodal_embedder\.embedding_projection)",
    },
    "gemma4": {
        "decoder": _DECODER,
        "vision_encoder": r"model\.vision_tower\.encoder\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.linear",
        "visual_connector": r"model\.(?:vision_tower\.patch_embedder\.input_proj|embed_vision\.embedding_projection)",
    },
}


def select_lora_targets(model):
    family = model.config.model_type
    if family not in _TARGETS:
        raise ValueError(f"No reviewed LoRA targets for model type {family}")
    patterns = _TARGETS[family]
    targets, counts = [], Counter()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        group = next(
            (key for key, pattern in patterns.items() if re.fullmatch(pattern, name)),
            None,
        )
        if group:
            targets.append(name)
            counts[group] += 1
        elif name != "lm_head" and "audio" not in name:
            raise ValueError(f"Unreviewed linear module in {family}: {name}")
    if any(not counts[group] for group in patterns):
        raise ValueError(
            f"Missing expected adapter component in {family}: {dict(counts)}"
        )
    return targets, dict(counts)


def verify_adapter_parameters(model):
    trainable = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Only LoRA adapter parameters may be trainable")
    if any(value.dtype != torch.float32 for _, value in trainable):
        raise ValueError("LoRA adapter parameters must remain FP32")
    return sum(value.numel() for _, value in trainable)
