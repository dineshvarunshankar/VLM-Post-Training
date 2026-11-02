"""Export the selected LoRA model as independently loadable merged weights."""

import gc
import hashlib
import tempfile
from pathlib import Path

import torch
from scripts.train.storage.cloud_models import digest_file, local_file
from scripts.train.shared.attention_compat import cpu_export_attention
from scripts.train.shared.data import atomic_json, read_json


def file_records(directory):
    records = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Export files must not be symlinks")
        if path.is_file():
            records[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size,
                "sha256": digest_file(path),
            }
    return records


def verify_export(run):
    run = Path(run)
    merged = run / "merged"
    report = read_json(local_file(merged, "export_manifest.json"))
    if (report.get("kind") != "merged_lora" or not report.get("reload_verified")
            or not report.get("adapter_completeness_verified")):
        raise ValueError("Run has no verified standalone merged model")
    if (merged / "adapter_config.json").exists():
        raise ValueError(
            "Deployment directory must not contain an adapter configuration"
        )
    expected = report["files"]
    actual = {str(p.relative_to(merged)) for p in merged.rglob("*") if p.is_file()}
    if actual != set(expected) | {"export_manifest.json"}:
        raise ValueError("Merged export contains missing or unexpected files")
    for relative, record in expected.items():
        path = local_file(merged, relative)
        if (
            path.stat().st_size != record["bytes"]
            or digest_file(path) != record["sha256"]
        ):
            raise ValueError(f"Merged model verification failed: {relative}")
    if file_records(run / "adapter") != report["adapter_files"]:
        raise ValueError("Adapter differs from the verified merged export")
    return report



def _tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _expected_lora_merges(model):
    """Fingerprint standard Linear merges one layer at a time, before mutation."""
    from peft.tuners.lora.layer import Linear, LoraLayer

    expected = []
    if any(config.modules_to_save for config in model.peft_config.values()):
        raise ValueError('Export merge verification does not support modules_to_save')
    for name, layer in model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        active = [key for key in layer.active_adapters if key in layer.lora_A]
        if (not isinstance(layer, Linear) or layer.fan_in_fan_out or layer.lora_variant
                or layer.merged or layer.disable_adapters or len(active) != 1):
            raise ValueError(f'Unsupported adapter merge for verification: {name}')
        adapter = active[0]
        if layer.lora_bias[adapter]:
            raise ValueError('Export merge verification does not support LoRA bias')
        a, b = layer.lora_A[adapter].weight, layer.lora_B[adapter].weight
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError('Export merge verification requires FP32 LoRA adapters')
        base = layer.get_base_layer()
        # PEFT safe_merge rounds the delta to base dtype BEFORE adding it.
        # This differs from a FP32 add followed by one final BF16 cast.
        with torch.no_grad():
            delta = (b @ a) * layer.scaling[adapter]
            value = base.weight.detach().clone()
            value += delta.to(value.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f'Non-finite expected merged weight: {name}')
            expected.append((name, base, _tensor_digest(value)))
            del value, delta
    if not expected:
        raise ValueError('Export merge verification found no supported adapters')
    return expected


def _logit_comparison(actual, reference):
    close = torch.isclose(actual, reference, rtol=0.02, atol=0.1)
    return {'rtol': 0.02, 'atol': 0.1, 'within_tolerance': bool(close.all()),
            'mismatched_elements': int((~close).sum()), 'elements': close.numel(),
            'max_abs_error': float((actual - reference).abs().max())}


def export_merged(model, processor, run, selection):
    """Called on rank zero after the trainer restores its best checkpoint."""
    run = Path(run)
    destination = run / "merged"
    if destination.exists():
        raise ValueError(
            "Merged output already exists; use a new run instead of overwriting it"
        )
    if not (run / "adapter/adapter_config.json").is_file():
        raise ValueError("Save the selected adapter before exporting merged weights")
    from peft.utils.save_and_load import load_peft_weights
    from scripts.train.shared.peft_compat import validate_adapter_state

    adapter_tensors = validate_adapter_state(
        model, load_peft_weights(str(run / "adapter"), device="cpu", local_files_only=True)
    )
    model.eval()
    inputs = processor.tokenizer("Return JSON.", return_tensors="pt")
    device = next(model.parameters()).device
    with torch.inference_mode(), cpu_export_attention(model):
        reference = (
            model(**{k: v.to(device) for k, v in inputs.items()})
            .logits[:, -1]
            .float()
            .cpu()
        )
    expected_merges = _expected_lora_merges(model)
    merged = model.merge_and_unload(safe_merge=True)
    merged_parameters = {id(parameter) for parameter in merged.parameters()}
    for name, layer, expected_digest in expected_merges:
        if id(layer.weight) not in merged_parameters or _tensor_digest(layer.weight) != expected_digest:
            raise ValueError(f'Merged weight differs from verified LoRA arithmetic: {name}')
    verified_merge_count = len(expected_merges)
    del expected_merges
    merged.to(device="cpu")
    merged.eval()
    merged.config.use_cache = True
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise ValueError("LoRA parameters remain after merging")
    with torch.inference_mode(), cpu_export_attention(merged):
        merged_reference = merged(**inputs).logits[:, -1].float().cpu()
    if not torch.isfinite(reference).all() or not torch.isfinite(merged_reference).all():
        raise ValueError("Non-finite export verification logits")
    adapter_drift = _logit_comparison(merged_reference, reference)
    with tempfile.TemporaryDirectory(prefix=".exporting-", dir=run) as temporary:
        staging = Path(temporary) / "merged"
        merged.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
        processor.save_pretrained(staging)
        if (staging / "adapter_config.json").exists():
            raise ValueError("Merged model was saved as an adapter")
        # Reload solely from the saved folder; no base-model or HF lookup is permitted.
        reloaded, info = type(merged).from_pretrained(
            staging,
            local_files_only=True,
            trust_remote_code=False,
            dtype=next(merged.parameters()).dtype,
            device_map="cpu",
            output_loading_info=True,
            attn_implementation=merged.config._attn_implementation,
        )
        if any(
            info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(
                f"Merged model reload reported incompatible weights: {info}"
            )
        reloaded.eval()
        with torch.inference_mode(), cpu_export_attention(reloaded):
            actual = reloaded(**inputs).logits[:, -1].float()
        # Isolate serialization from BF16 adapter-to-merge rounding. Both
        # forwards use the same merged weights, CPU kernels and attention mode.
        torch.testing.assert_close(actual, merged_reference, rtol=0.02, atol=0.1)
        reload_comparison = _logit_comparison(actual, merged_reference)
        del reloaded
        gc.collect()
        atomic_json(
            staging / "export_manifest.json",
            {
                "kind": "merged_lora",
                "reload_verified": True,
                "adapter_completeness_verified": True,
                "adapter_tensor_count": adapter_tensors,
                "verification": "exact LoRA merge arithmetic and offline merged-model text-logit comparison",
                "merge_weights_verified": True,
                "merge_weight_count": verified_merge_count,
                "adapter_to_merged_logits": adapter_drift,
                "merged_to_reload_logits": reload_comparison,
                "selection": selection,
                "adapter_files": file_records(run / "adapter"),
                "files": file_records(staging),
            },
        )
        staging.rename(destination)
    return destination


def export_saved_run(run, weights_root=None):
    """Retry an interrupted export from its saved selected adapter."""
    import transformers
    from scripts.train.storage.cloud_models import verify_local_source
    from peft import PeftModel

    run = Path(run).resolve()
    metadata = read_json(run / "run.json")
    profile = metadata["model"]
    source = (
        Path(weights_root).expanduser() / profile["repository"] / profile["revision"]
        if weights_root
        else Path(profile["local_path"])
    )
    verify_local_source(source, profile)
    options = {"local_files_only": True, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **options)
    if not config.architectures or len(config.architectures) != 1:
        raise ValueError("Expected one supported model architecture")
    model_class = getattr(transformers, config.architectures[0])
    base = model_class.from_pretrained(
        source, dtype=torch.bfloat16, device_map="cpu", **options
    )
    model = PeftModel.from_pretrained(base, run / "adapter", local_files_only=True)
    processor = transformers.AutoProcessor.from_pretrained(source, **options)
    return export_merged(
        model, processor, run, read_json(run / "checkpoint_selection.json")
    )
