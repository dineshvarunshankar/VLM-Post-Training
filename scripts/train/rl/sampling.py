"""Epoch task sampling with TRL's response-group and rollout repetition order."""

import math
from torch.utils.data import Sampler
from scripts.train.shared.mixture import EpochMixtureSampler


class RLQuestionSampler(Sampler):
    def __init__(self, plan, *, seed, shuffle, resample, questions_per_rollout,
                 num_generations, steps_per_generation):
        for value in (questions_per_rollout, num_generations, steps_per_generation):
            if type(value) is not int or value <= 0:
                raise ValueError('RL sampler batch values must be positive integers')
        self.questions = EpochMixtureSampler(plan, seed=seed, shuffle=shuffle, resample=resample)
        self.batch = questions_per_rollout
        self.generations = num_generations
        self.repeats = steps_per_generation
        self.padding_questions = (-len(self.questions)) % self.batch

    def set_epoch(self, epoch):
        self.questions.set_epoch(epoch)

    def __len__(self):
        return math.ceil(len(self.questions) / self.batch) * self.batch * self.generations * self.repeats

    def __iter__(self):
        selected = list(self.questions)
        # Explicit final-batch padding retains every medical example. Report this
        # count separately; it is not a difficulty-based sampling policy.
        selected += [selected[i % len(selected)] for i in range(self.padding_questions)]
        for start in range(0, len(selected), self.batch):
            chunk = selected[start:start + self.batch]
            for _ in range(self.repeats):
                for index in chunk:
                    yield from [index] * self.generations
