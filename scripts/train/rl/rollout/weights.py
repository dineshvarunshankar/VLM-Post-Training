"""Stream LoRA-combined rollout weights without changing the training model."""

from types import MethodType

import torch
from peft import PeftModel
from peft.tuners.lora.layer import Linear, LoraLayer
from torch.distributed.tensor import DTensor
from torch.distributed.fsdp import FSDPModule


def _full_tensor(parameter):
    value = parameter.detach()
    return value.full_tensor() if isinstance(value, DTensor) else value


def _weight_name(name):
    return (name.removeprefix('base_model.model.')
            .replace('_checkpoint_wrapped_module.', '')
            .replace('.base_layer', ''))


def _checkpoint_weight_name(name, model_type):
    name = _weight_name(name)
    if model_type == 'gemma4_unified':
        # Transformers 5.21 combines the legacy vision embedder and projection
        # into embed_vision. vLLM 0.30 consumes the original checkpoint layout.
        # Reverse only the six Gemma Unified load renames, at module boundaries.
        for module in ('patch_ln1', 'patch_dense', 'patch_ln2', 'pos_embedding', 'pos_norm'):
            prefix = 'model.embed_vision.' + module
            if name == prefix or name.startswith(prefix + '.'):
                return 'model.vision_embedder.' + name.removeprefix('model.embed_vision.')
        prefix = 'model.embed_vision.multimodal_embedder.embedding_projection'
        if name == prefix or name.startswith(prefix + '.'):
            return 'model.embed_vision.embedding_projection' + name[len(prefix):]
    return name


def iter_policy_weights(model):
    """Gather and combine one parameter at a time on every participating rank.

    Supports the approved standard Linear LoRA adapters, including FSDP2 DTensors.
    The training base, adapters, and optimizer state remain untouched. Inference
    receives combined weights in the base dtype; numerical output parity still
    needs testing because merged BF16 inference rounds adapter updates.
    """
    if not isinstance(model, PeftModel):
        raise ValueError('Expected a PEFT LoRA policy')
    if any(config.modules_to_save for config in model.peft_config.values()):
        raise ValueError('modules_to_save is not supported by this weight exporter')
    # A no-grad forward may leave BF16 compute parameters registered, especially
    # at the FSDP2 root. Restore sharded master parameters before reading LoRA
    # weights: casting the compute copies would lose the original FP32 values.
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    layers = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        if (not isinstance(layer, Linear) or layer.fan_in_fan_out
                or layer.lora_variant or layer.merged or layer.disable_adapters):
            raise ValueError(f'Expected an active, unmerged standard Linear LoRA layer: {name}')
        active = [key for key in layer.active_adapters if key in layer.lora_A]
        if len(active) != 1:
            raise ValueError(f'Expected one active policy adapter in {name}')
        adapter = active[0]
        if layer.lora_B[adapter].bias is not None:
            raise ValueError('LoRA bias updates are not configured for this recipe')
        layers[name + '.base_layer.weight'] = (layer, adapter)
    if not layers:
        raise ValueError('Policy has no supported LoRA layers')

    model_type = getattr(getattr(model.get_base_model(), 'config', None), 'model_type', None)
    visited = set()
    for name, parameter in model.named_parameters():
        if any(part.startswith('lora_') for part in name.split('.')):
            continue
        output_name = _checkpoint_weight_name(name, model_type)
        if output_name in visited:
            raise ValueError(f'Duplicate rollout weight name: {output_name}')
        visited.add(output_name)
        value = _full_tensor(parameter)
        if name in layers:
            layer, adapter = layers[name]
            a = _full_tensor(layer.lora_A[adapter].weight)
            b = _full_tensor(layer.lora_B[adapter].weight)
            if value.dtype != torch.bfloat16 or a.dtype != torch.float32 or b.dtype != torch.float32:
                raise ValueError(
                    f'Expected BF16 base weights and FP32 LoRA adapters: {name}; '
                    f'base={value.dtype}, A={a.dtype}, B={b.dtype}'
                )
            value = torch.addmm(
                value.float(), b, a, alpha=float(layer.scaling[adapter])
            ).to(value.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite combined rollout weight: {output_name}')
            del a, b
        yield output_name, value


def install_policy_weight_export(generation):
    """Replace only this TRL helper's parameter iterator; retain its transport."""
    backend = generation._dist
    if backend.is_zero3 or (backend.is_fsdp and backend.fsdp_version != 2):
        raise ValueError('This rollout weight exporter supports FSDP2 or unsharded models')
    if generation.mode != 'server':
        raise ValueError('This rollout weight exporter requires vLLM server mode')

    def iterator(helper):
        return iter_policy_weights(helper.model)

    generation._iter_named_params = MethodType(iterator, generation)
    generation._weight_metadata = None
