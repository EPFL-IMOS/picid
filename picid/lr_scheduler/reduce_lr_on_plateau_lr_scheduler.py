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
from torch.optim import Optimizer

from .lr_scheduler import LearningRateScheduler


class ReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Reduce the learning rate when a metric has stopped improving.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    lr : float
        Initial learning rate.
    patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    factor : float
        Multiplicative factor used when reducing the learning rate.
    min_lr : float, default=1e-8
        Lower bound for the learning rate.
    apply_init_lr_to_scheduler : bool, default=True
        Whether to immediately apply the initial learning rate to the optimizer.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr: float,
        patience: int = 1,
        factor: float = 0.3,
        min_lr: float = 1e-8,
        apply_init_lr_to_scheduler: bool = True,
    ):
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, lr)
        self.optimizer = optimizer
        self.lr = lr
        self.best_val_loss = float("inf")
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.num_bad_epochs = 0

        if apply_init_lr_to_scheduler:
            self.set_lr(self.optimizer, lr)

    def step(self, val_loss: float):
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.lr = max(self.lr * self.factor, self.min_lr)
            self.set_lr(self.optimizer, self.lr)
            self.num_bad_epochs = 0

        return self.lr
