import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupLinearSchedule(_LRScheduler):
    def __init__(self, optimiser, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearSchedule, self).__init__(optimiser)

    def get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            remaining_steps = self.total_steps - step
            return [base_lr * remaining_steps / (self.total_steps - self.warmup_steps) for base_lr in self.base_lrs]
