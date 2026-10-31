"""Deterministic epoch sampling over an immutable pool of training examples."""

import random

from torch.utils.data import Sampler


class EpochMixtureSampler(Sampler):
    def __init__(self, plan, *, seed, shuffle=True, resample=True):
        self.plan = plan
        self.seed = seed
        self.shuffle = shuffle
        self.resample = resample
        self.epoch = 0
        self.size = len(plan["fixed"]) + sum(g["count"] for g in plan["groups"])
        if self.size == 0:
            raise ValueError("Configured mixture has no eligible training examples")
        all_indices = list(plan["fixed"])
        for group in plan["groups"]:
            if len(group["indices"]) != len(group["values"]):
                raise ValueError("Candidate indices and values must have equal lengths")
            if not group["indices"] or group["count"] < 1:
                raise ValueError("Sampling groups must be nonempty")
            if not group["replace"] and group["count"] > len(group["indices"]):
                raise ValueError("Sampling quota exceeds available distinct examples")
            all_indices.extend(group["indices"])
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Sampling groups and fixed examples must not overlap")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.size

    def __iter__(self):
        rng = random.Random(self.seed + (self.epoch if self.resample else 0))
        selected = list(self.plan["fixed"])
        for group in self.plan["groups"]:
            indices, count = group["indices"], group["count"]
            chosen = []
            if group["balanced"] and count >= 2:
                positives = [i for i, v in zip(indices, group["values"]) if v == 1]
                negatives = [i for i, v in zip(indices, group["values"]) if v == 0]
                if positives and negatives:
                    chosen = [rng.choice(positives), rng.choice(negatives)]
            remaining = (
                indices if group["replace"] else [i for i in indices if i not in chosen]
            )
            draw = rng.choices if group["replace"] else rng.sample
            chosen.extend(draw(remaining, k=count - len(chosen)))
            selected.extend(chosen)
        if self.shuffle:
            random.Random(self.seed + self.epoch + 1_000_003).shuffle(selected)
        return iter(selected)
