"""Collectively reject nonfinite gradients before SFT optimizer updates."""
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


@torch.no_grad()
def require_finite_gradients(parameters, gradient_norm, *, device):
    # Inspect local FSDP shards; do not gather full gradients or change them.
    checks = [torch.isfinite(_local(torch.as_tensor(gradient_norm, device=device))).all()]
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = _local(parameter.grad)
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            checks.append(torch.isfinite(gradient).all().to(device))
    invalid = (~torch.stack(checks).all()).to(dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    if invalid.item():
        raise FloatingPointError(
            "SFT optimizer update blocked on all ranks: nonfinite gradient or gradient norm"
        )


class FiniteGradientMixin:
    def _clip_grad_norm(self, model):
        # Trainer calls this after accumulation/synchronization, before optimizer.step.
        # Keep its native clipping (including mixed FSDP meshes) and pre-clip norm.
        norm = super()._clip_grad_norm(model)
        require_finite_gradients(model.parameters(), norm, device=self.accelerator.device)
        return norm
