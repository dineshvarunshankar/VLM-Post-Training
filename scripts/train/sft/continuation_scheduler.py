"""A checkpoint-persisted cosine tail anchored after optimizer restoration."""
from copy import deepcopy
import math

from torch.optim.lr_scheduler import LRScheduler
from scripts.train.sft.continuation import validate_policy


class CosineTailScheduler(LRScheduler):
    def __init__(self, optimizer, end_step, policy):
        validate_policy(policy)
        self.end_step = end_step
        self.policy = deepcopy(policy)
        self.anchor_step = None
        self.anchor_lrs = None
        self.restored = False
        super().__init__(optimizer)

    def get_lr(self):
        if not self.restored:
            return [group['lr'] for group in self.optimizer.param_groups]
        progress = min(1., max(0., (self.last_epoch - self.anchor_step) / (self.end_step - self.anchor_step)))
        factor = (1 + math.cos(math.pi * progress)) / 2
        return [lr * factor for lr in self.anchor_lrs]

    def state_dict(self):
        return {'continuation_schema': 1, 'end_step': self.end_step, 'policy': deepcopy(self.policy),
                'anchor_step': self.anchor_step, 'anchor_lrs': self.anchor_lrs,
                'last_epoch': self.last_epoch, '_step_count': self._step_count,
                'base_lrs': self.base_lrs, '_last_lr': self._last_lr}

    def load_state_dict(self, state):
        state = deepcopy(state)
        step = state.get('last_epoch')
        if type(step) is not int or not 0 < step < self.end_step:
            raise ValueError('Continuation horizon must follow the saved scheduler step')
        if 'continuation_schema' in state:
            if (state['continuation_schema'] != 1 or state.get('end_step') != self.end_step
                    or state.get('policy') != self.policy):
                raise ValueError('Saved continuation policy or horizon differs')
            anchor_step, lrs = state.get('anchor_step'), state.get('anchor_lrs')
            if type(anchor_step) is not int or not 0 < anchor_step <= step:
                raise ValueError('Invalid saved continuation anchor step')
        else:
            anchor_step = step
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            override = self.policy['start_learning_rate']
            if override is not None:
                if any(lr != 0 for lr in lrs):
                    raise ValueError('Explicit continuation LR requires an exhausted zero-LR schedule')
                lrs = [override] * len(lrs)
        if (not isinstance(lrs, list) or len(lrs) != len(self.optimizer.param_groups)
                or any(type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0 for lr in lrs)):
            raise ValueError('Continuation requires positive finite anchor LRs; exhausted schedules need an explicit LR')
        self.anchor_step, self.anchor_lrs = anchor_step, lrs
        self.last_epoch = step
        self._step_count = state.get('_step_count', step + 1)
        self.base_lrs = state['base_lrs']
        self.restored = True
        self._last_lr = self.get_lr()
        if 'continuation_schema' in state:
            saved_lrs = [group['lr'] for group in self.optimizer.param_groups]
            if any(not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0)
                   for actual, expected in zip(saved_lrs, self._last_lr)):
                raise ValueError('Saved optimizer LR differs from continuation schedule')
        for group, lr in zip(self.optimizer.param_groups, self._last_lr):
            group['lr'] = lr
