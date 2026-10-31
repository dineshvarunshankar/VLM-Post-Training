"""Export the selected LoRA model as independently loadable merged weights."""

import gc
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
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise ValueError("LoRA parameters remain after merging")
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
        # BF16 merge and CPU/GPU kernels can differ slightly; this is a smoke check,
        # not a replacement for the held-out multimodal evaluation.
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.1)
        del reloaded
        gc.collect()
        atomic_json(
            staging / "export_manifest.json",
            {
                "kind": "merged_lora",
                "reload_verified": True,
                "adapter_completeness_verified": True,
                "adapter_tensor_count": adapter_tensors,
                "verification": "offline reload and text-logit comparison",
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
