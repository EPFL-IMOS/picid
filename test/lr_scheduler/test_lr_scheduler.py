"""Tests for picid.lr_scheduler.lr_scheduler base class."""
import pytest
import torch
from torch import nn
from torch.optim import SGD

from picid.lr_scheduler.lr_scheduler import LearningRateScheduler


class TestLearningRateScheduler:
    def test_set_lr_updates_all_param_groups(self):
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        LearningRateScheduler.set_lr(optimizer, 0.001)
        for g in optimizer.param_groups:
            assert g["lr"] == 0.001

    def test_get_lr_returns_first_group_lr(self):
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.05)
        scheduler = LearningRateScheduler(optimizer, 0.05)
        assert scheduler.get_lr() == 0.05

    def test_step_raises_not_implemented(self):
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = LearningRateScheduler(optimizer, 0.01)
        with pytest.raises(NotImplementedError):
            scheduler.step()
