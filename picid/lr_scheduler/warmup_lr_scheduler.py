# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from typing import Optional
from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class WarmupLRScheduler(LearningRateScheduler):
    """
    Warm up the optimizer learning rate over a fixed number of steps.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer whose parameter groups will be updated in place.
    init_lr : float
        Learning rate used before warmup starts.
    peak_lr : float
        Target learning rate reached at the end of warmup.
    warmup_steps : int
        Number of scheduler steps used for linear warmup.
    apply_init_lr_to_scheduler : bool, default=True
        Whether to apply ``init_lr`` immediately when the scheduler is created.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        warmup_steps: int,
        apply_init_lr_to_scheduler: bool = True,
    ) -> None:
        super(WarmupLRScheduler, self).__init__(optimizer, init_lr)
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.warmup_steps = int(warmup_steps)
        self.update_steps = 0
        self.lr = init_lr
        self.warmup_rate = (
            0.0 if self.warmup_steps == 0 else (peak_lr - init_lr) / self.warmup_steps
        )

        if apply_init_lr_to_scheduler:
            self.set_lr(self.optimizer, init_lr)

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        # increment first so that the Nth call reaches peak_lr when warmup_steps == N
        self.update_steps += 1
        if self.warmup_steps > 0 and self.update_steps <= self.warmup_steps:
            # exact linear interpolation with clamping on the last warmup step
            progress = self.update_steps / self.warmup_steps
            lr = self.init_lr + (self.peak_lr - self.init_lr) * progress
            if self.update_steps == self.warmup_steps:
                lr = self.peak_lr
            self.set_lr(self.optimizer, lr)
            self.lr = lr
