"""Teacher-forced medical JSON diagnostics; these never change the objective."""
import json
import re

import torch
import torch.nn.functional as F


def target_groups(suffix, answer, offsets, *, medical):
    """0: excluded, 1: value-bearing token, 2: other JSON tokens.

    A tokenizer can merge a value with punctuation. Such a token belongs to
    the value group; measurements are token accuracy, not decoded label F1.
    """
    if not medical:
        return [0] * len(offsets)
    values = json.loads(answer)
    if not isinstance(values, dict) or not values or any(type(v) is not int or v not in (0, 1) for v in values.values()):
        raise ValueError('Medical diagnostics require a nonempty flat binary JSON answer')
    start = suffix.rfind(answer)
    if start < 0:
        raise ValueError('Final medical answer is absent from the assistant target')
    spans = [(start + m.start(1), start + m.end(1)) for m in re.finditer(
        r'"(?:\\.|[^"\\])*"\s*:\s*([01])(?=\s*[,}])', answer)]
    if len(spans) != len(values):
        raise ValueError('Cannot locate every medical value in the JSON target')
    result = []
    for left, right in offsets:
        if right <= left:
            result.append(0)
        elif any(left < end and right > begin for begin, end in spans):
            result.append(1)
        elif left < start + len(answer) and right > start:
            result.append(2)
        else:
            result.append(0)
    return result


@torch.no_grad()
def batch_statistics(outputs, labels, groups):
    """Return per-example sums/correct/counts for Accelerate's tail-aware gather.

    Prefer TRL fused per-token outputs. Older runtimes expose logits instead;
    process only selected positions in small chunks without a full fp32 copy.
    """
    target = labels[:, 1:]
    groups = groups[:, 1:].to(target.device)
    valid = target != -100
    stats = torch.zeros((len(labels), 6), dtype=torch.float64, device=target.device)
    log_probs = getattr(outputs, 'log_probs', None)
    if log_probs is not None:
        if log_probs.shape != target.shape or outputs.is_top1.shape != target.shape:
            raise ValueError('Unsupported fused token alignment for medical diagnostics')
        losses = -log_probs.detach().to(target.device)
        correct = outputs.is_top1.detach().to(target.device)
        valid = valid & outputs.label_mask.to(target.device)
        for group in (1, 2):
            mask = valid & (groups == group)
            offset = (group - 1) * 3
            stats[:, offset] = losses.masked_fill(~mask, 0).double().sum(1)
            stats[:, offset + 1] = (correct & mask).sum(1)
            stats[:, offset + 2] = mask.sum(1)
        return stats
    logits = getattr(outputs, 'logits', None)
    if logits is None or logits.shape[:2] != labels.shape:
        raise ValueError('Model did not return aligned token outputs for medical diagnostics')
    for row in range(len(labels)):
        for group in (1, 2):
            positions = (valid[row] & (groups[row] == group)).nonzero().flatten()
            offset = (group - 1) * 3
            for chunk in positions.split(32):
                scores = logits[row, chunk.to(logits.device)].detach().float()
                truth = target[row, chunk].to(scores.device)
                stats[row, offset] += F.cross_entropy(scores, truth, reduction='sum').double().to(stats.device)
                stats[row, offset + 1] += (scores.argmax(-1) == truth).sum().to(stats.device)
            stats[row, offset + 2] = len(positions)
    return stats


def summarize(totals, prefix):
    metrics = {}
    for index, name in enumerate(('value', 'format')):
        loss, correct, count = totals[index * 3:index * 3 + 3].tolist()
        key = f'{prefix}_medical_{name}'
        metrics[key + '_tokens'] = int(count)
        if count:
            metrics[key + '_loss'] = loss / count
            metrics[key + '_accuracy'] = correct / count
    return metrics
