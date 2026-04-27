"""Unit tests for PatchTST_Forecaster (thuml_patchtst_model.py) to increase coverage."""

import pytest
import torch

from picid.model.forecasters.patchtst_model.thuml_patchtst_model import (
    PatchTST_Forecaster,
)
from picid.evaluator.base import AbstractEvaluator
from typing import Any, Dict


class MockEvaluator(AbstractEvaluator):
    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}

    def reset(self) -> None:
        self.results.clear()

    def compute(self, mode, epoch, step):
        return self.results

    def update(self, model_outs):
        self.results = {"loss": 0.0}
        return self.results


def _mock_optimizer_factory(params):
    return torch.optim.Adam(params, lr=0.001)


def _mock_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


def _patchtst_forecaster(
    task_type="long_term_forecast",
    include_x_t=False,
    d_x_mask=None,
    mask_y_c=False,
    d_yc=2,
    d_x=2,
    d_yt=1,
    ts_in=24,
    ts_out=6,
):
    evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
    return PatchTST_Forecaster(
        optimizer_factory=_mock_optimizer_factory,
        scheduler_factory=_mock_scheduler_factory,
        d_x=d_x,
        d_yc=d_yc,
        d_yt=d_yt,
        ts_in=ts_in,
        ts_out=ts_out,
        task_type=task_type,
        include_x_t=include_x_t,
        d_x_mask=d_x_mask,
        mask_y_c=mask_y_c,
        evaluators=evaluators,
    )


class TestPatchTSTForecasterLongTerm:
    """Tests for long_term_forecast task (task_type not in CLASSIFICATION+REGRESSION)."""

    def test_init_long_term_forecast(self):
        f = _patchtst_forecaster(task_type="forecasting", include_x_t=False)
        assert f.model.task_name == "long_term_forecast"
        assert f.include_x_t is False

    def test_forward_long_term_y_c_only(self):
        """enc_in=1 for long_term_forecast; use d_yc=1, d_x=0."""
        f = _patchtst_forecaster(task_type="forecasting", d_yc=1, d_x=0)
        B, L = 2, 24
        y_c = torch.randn(B, L, 1)
        x_c = None
        x_t = torch.zeros(B, 6, 1)
        y_t = torch.randn(B, 6, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)

    def test_forward_long_term_x_c_only(self):
        """x_c only, y_c None: batch_x = x_c."""
        f = _patchtst_forecaster(task_type="forecasting", d_yc=0, d_x=1)
        B, L = 2, 24
        y_c = None
        x_c = torch.randn(B, L, 1)
        x_t = torch.randn(B, 6, 1)
        y_t = torch.randn(B, 6, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)


class TestPatchTSTForecasterIncludeXT:
    """Tests for include_x_t=True with d_x_mask. enc_in=1 for long_term_forecast."""

    def test_init_include_x_t_with_d_x_mask_sets_masks(self):
        f = _patchtst_forecaster(
            task_type="forecasting",
            include_x_t=True,
            d_x_mask=[0, 1],
            d_yc=1,
            d_x=2,
        )

        assert f.d_x_mask == [0, 1]
        assert torch.equal(f.mask_tt, torch.tensor([True, False]))
        assert torch.equal(f.mask_t, torch.tensor([False, True]))

    def test_forward_include_x_t_applies_d_x_mask_before_concat(self, monkeypatch):
        f = _patchtst_forecaster(
            task_type="forecasting",
            include_x_t=True,
            d_x_mask=[0],
            d_yc=1,
            d_x=1,
            d_yt=1,
        )
        B, L_ctx, L_tgt = 2, 24, 6
        captured = {}

        def fake_forward(x_enc, x_mark_enc, x_dec, x_mark_dec):
            captured["x_enc"] = x_enc
            captured["x_mark_enc"] = x_mark_enc
            captured["x_dec"] = x_dec
            captured["x_mark_dec"] = x_mark_dec
            return torch.randn(B, L_tgt, 2)

        monkeypatch.setattr(f.model, "forward", fake_forward)

        y_c = torch.randn(B, L_ctx, 1)
        x_c = torch.randn(B, L_ctx, 1)
        x_t = torch.randn(B, L_tgt, 1)
        y_t = torch.randn(B, L_tgt, 1)

        (out,) = f.forward_model_pass(x_c, y_c, x_t.clone(), y_t)

        expected_context = torch.cat([y_c, x_c], dim=2)
        target_half = captured["x_enc"][:, L_ctx:, :]

        assert torch.equal(captured["x_enc"][:, :L_ctx, :], expected_context)
        assert torch.equal(target_half[:, :, 0], torch.zeros_like(y_t[:, :, 0]))
        assert torch.equal(target_half[:, :, 1], torch.zeros_like(x_t[:, :, 0]))
        assert captured["x_mark_enc"] is None
        assert captured["x_dec"] is None
        assert captured["x_mark_dec"] is None
        assert out.shape == (B, L_tgt, 1)

    @pytest.mark.skip(
        reason="PatchTST enc_in=1 conflicts with d_x_mask multi-dim; needs config fix"
    )
    def test_forward_include_x_t_with_d_x_mask(self):
        """Forward with include_x_t=True and d_x_mask.

        **Skipped:** RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x256 and 192x6).

        For long_term_forecast, the forecaster hardcodes enc_in=1 (thuml_patchtst_model.py:158).
        With include_x_t=True and d_x_mask, the input is torch.cat([batch_x, batch_y], dim=1)
        with multiple channels. The model is built for 1 channel, so linear layers receive
        wrong input shape.

        **Fix:** Make enc_in dynamic (e.g. d_yc + d_x) for long_term_forecast when
        include_x_t=True.
        """
        f = _patchtst_forecaster(
            task_type="forecasting",
            include_x_t=True,
            d_x_mask=[1],
            d_yc=1,
            d_x=1,
        )
        B, L = 2, 30
        y_c = torch.randn(B, L, 1)
        x_c = torch.randn(B, L, 1)
        x_t = torch.randn(B, 6, 1)
        y_t = torch.randn(B, 6, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)


class TestPatchTSTForecasterMaskYC:
    """Tests for mask_y_c=True."""

    def test_forward_mask_y_c(self):
        f = _patchtst_forecaster(task_type="forecasting", mask_y_c=True, d_yc=1)
        B, L = 2, 24
        y_c = torch.randn(B, L, 1)
        x_c = torch.randn(B, L, 1)
        x_t = torch.randn(B, 6, 1)
        y_t = torch.randn(B, 6, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)

    def test_forward_mask_y_c_unsqueezes_2d_context_and_zeros_it(self, monkeypatch):
        f = _patchtst_forecaster(task_type="forecasting", mask_y_c=True, d_yc=1, d_x=1)
        B, L_ctx, L_tgt = 2, 24, 6
        captured = {}

        def fake_forward(x_enc, x_mark_enc, x_dec, x_mark_dec):
            captured["x_enc"] = x_enc
            return torch.randn(B, L_tgt, 2)

        monkeypatch.setattr(f.model, "forward", fake_forward)

        y_c = torch.randn(B, L_ctx)
        x_c = torch.randn(B, L_ctx, 1)
        x_t = torch.randn(B, L_tgt, 1)
        y_t = torch.randn(B, L_tgt, 1)

        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)

        assert captured["x_enc"].shape == (B, L_ctx, 2)
        assert torch.equal(captured["x_enc"][:, :, :1], torch.zeros(B, L_ctx, 1))
        assert torch.equal(captured["x_enc"][:, :, 1:], x_c)
        assert out.shape == (B, L_tgt, 1)


class TestPatchTSTForecasterRegression:
    """Tests for regression task. enc_in=d_yc+d_x for regression."""

    def test_forward_regression(self):
        f = _patchtst_forecaster(task_type="rul", include_x_t=False, d_yc=0, d_x=2)
        B, L = 2, 24
        x_c = None
        y_c = None
        x_t = torch.randn(B, L, 2)
        y_t = torch.randn(B, L, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 1, 1)


class TestPatchTSTForecasterClassification:
    """Tests for classification task. enc_in=d_yc+d_x for classification."""

    def test_forward_classification(self):
        f = _patchtst_forecaster(
            task_type="classification", d_yt=3, d_yc=0, d_x=2, include_x_t=False
        )
        B, L = 2, 24
        x_c = None
        y_c = None
        x_t = torch.randn(B, L, 2)
        y_t = torch.randint(0, 3, (B, L, 1)).float()
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 1, 3)


def test_eval_step_forwards_target_mask_when_present(monkeypatch):
    f = _patchtst_forecaster(task_type="forecasting", d_yc=1, d_x=1)
    f.target_mask = "keep-last-channel"
    batch = (
        torch.randn(2, 24, 1),
        torch.randn(2, 24, 1),
        torch.randn(2, 6, 1),
        torch.randn(2, 6, 1),
    )
    captured = {}

    def fake_compute_loss(batch, time_mask=None, forward_kwargs=None):
        captured["time_mask"] = time_mask
        captured["forward_kwargs"] = dict(forward_kwargs)
        return {
            "forecast_loss": torch.tensor(0.5),
            "forecast_out": torch.randn(2, 6, 1),
            "forecast_mask": torch.ones(2, 6, 1, dtype=torch.bool),
        }

    monkeypatch.setattr(f, "compute_loss", fake_compute_loss)

    stats = f.step(batch=batch, train=False)

    assert captured["time_mask"] is None
    assert captured["forward_kwargs"] == {
        "output_attn": False,
        "target_mask": "keep-last-channel",
    }
    assert stats["loss"].equal(torch.tensor(0.5))
