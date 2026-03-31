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

from torch.optim import Optimizer

from typing import Optional

from .lr_scheduler import LearningRateScheduler
from .reduce_lr_on_plateau_lr_scheduler import ReduceLROnPlateauScheduler
from .warmup_lr_scheduler import WarmupLRScheduler


class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Warm up the learning rate and then reduce it on plateau.

    Parameters
    ----------
    optimizer : Optimizer
        Wrapped optimizer.
    init_lr : float
        Initial learning rate.
    peak_lr : float
        Maximum learning rate.
    warmup_steps : int
        Warmup the learning rate linearly for the first N updates.
    patience : int
        Number of epochs with no improvement after which learning rate will be reduced.
    factor : float
        Multiplicative factor used when reducing the learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        init_lr: float,
        peak_lr: float,
        warmup_steps: int,
        patience: int = 1,
        factor: float = 0.3,
    ) -> None:
        super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, init_lr)
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        self.warmup_rate = (
            (peak_lr - init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        )
        self.schedulers = [
            WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
                apply_init_lr_to_scheduler=False,
            ),
            ReduceLROnPlateauScheduler(
                optimizer=optimizer,
                lr=peak_lr,
                patience=patience,
                factor=factor,
                apply_init_lr_to_scheduler=False,
            ),
        ]

        self.set_lr(optimizer, init_lr)

    def load_state_dict(self, state_dict):
        self.schedulers[0].load_state_dict(state_dict["warmup"])
        self.schedulers[1].load_state_dict(state_dict["reduce_lr_on_plateau"])
        self.update_steps = state_dict["update_steps"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.warmup_rate = state_dict["warmup_rate"]

    def state_dict(self):
        return {
            "update_steps": self.update_steps,
            "warmup": self.schedulers[0].state_dict(),
            "reduce_lr_on_plateau": self.schedulers[1].state_dict(),
            "warmup_steps": self.warmup_steps,
            "warmup_rate": self.warmup_rate,
        }

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss: Optional[float] = None, is_end_epoch=False):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1 and is_end_epoch:
            self.schedulers[1].step(val_loss)

        self.update_steps += 1

        # Keep for debugging purposes
        # lr = self.get_lr()
        # lr = self.optimizer.param_groups[0]["lr"]

        # if stage == 0:
        #     print(f"\n[0] step: {self.update_steps}, lr: {lr}")
        # elif stage == 1 and is_end_epoch:
        #     print(f"\n[1] step: {self.update_steps}, lr: {lr}")
