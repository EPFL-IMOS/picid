"""Tests for picid.lr_scheduler.lr_scheduler base class."""
import pytest
import torch
from torch import nn
from torch.optim import SGD

from picid.lr_scheduler.lr_scheduler import LearningRateScheduler
from picid.lr_scheduler.warmup_lr_scheduler import WarmupLRScheduler


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


class TestWarmupLRScheduler:
    def test_warmup_linear_interpolation(self):
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = WarmupLRScheduler(
            optimizer, init_lr=0.01, peak_lr=0.1, warmup_steps=5
        )
        for _ in range(5):
            scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    def test_warmup_zero_steps_stays_at_init(self):
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = WarmupLRScheduler(
            optimizer, init_lr=0.01, peak_lr=0.1, warmup_steps=0
        )
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    def test_apply_init_lr_to_scheduler_false(self):
        model = nn.Linear(2, 2)
        optimizer = SGD(model.parameters(), lr=0.5)
        WarmupLRScheduler(
            optimizer, init_lr=0.01, peak_lr=0.1, warmup_steps=3,
            apply_init_lr_to_scheduler=False,
        )
        assert optimizer.param_groups[0]["lr"] == 0.5
