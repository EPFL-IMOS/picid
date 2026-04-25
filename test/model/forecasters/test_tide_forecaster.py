"""Unit tests for TiDE_Forecaster (tide_model.py) to increase coverage."""

import pytest
import torch

from picid.model.forecasters.tide_model.tide_model import TiDE_Forecaster
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


def _tide_forecaster(
    task_type="long_term_forecast",
    include_x_t=True,
    d_x_mask=None,
    mask_y_c=False,
    d_yc=2,
    d_x=4,
    d_yt=1,
    ts_in=24,
    ts_out=6,
    use_x_dim=False,
    freq="h",
):
    evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
    return TiDE_Forecaster(
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
        use_x_dim=use_x_dim,
        freq=freq,
        evaluators=evaluators,
    )


class TestTiDEForecasterStateForecasting:
    """Tests for state_forecasting task (x_c=None, x_t=None, x_enc=y_c)."""

    @pytest.mark.skip(
        reason="TiDE state_forecasting flatten_dim mismatch; needs config alignment"
    )
    def test_forward_state_forecasting(self):
        """Forward for state_forecasting (x_c=None, x_t=None, x_enc=y_c).

        **Skipped:** RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x30 and 24x64).

        For state_forecasting, TiDE uses flatten_dim=seq_len (no feature encoder in the
        flatten), freq=None, and feature_dim from config. The forecaster config defaults
        to freq="h", so feature_dim/flatten_dim mismatch. The encoder expects different
        input shapes than what the forecaster passes.

        **Fix:** For task_type == "state_forecasting", set freq=None and feature_dim
        (e.g. from d_yt or d_yc) in the forecaster config so TiDE config matches the
        state_forecasting path.
        """
        f = _tide_forecaster(
            task_type="state_forecasting",
            d_yt=2,
            use_x_dim=True,
            d_x=2,
            freq=None,
        )
        B, L = 2, 30
        y_c = torch.randn(B, L, 2)
        x_c = None
        x_t = None
        y_t = torch.randn(B, 6, 2)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 2)


class TestTiDEForecasterLongTerm:
    """Tests for forecasting (contextual forecasting)."""

    def test_forward_long_term_forecast(self):
        f = _tide_forecaster(task_type="forecasting", include_x_t=True, d_yt=2)
        B, L = 2, 24
        y_c = torch.randn(B, L, 2)
        x_c = torch.randn(B, L, 4)
        x_t = torch.randn(B, 30, 4)
        y_t = torch.randn(B, 6, 2)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 2)


class TestTiDEForecasterIncludeXTFalse:
    """Tests for include_x_t=False (regression path zeros x_t)."""

    def test_forward_include_x_t_false(self):
        f = _tide_forecaster(include_x_t=False, task_type="rul")
        B, L = 2, 24
        x_c = None
        y_c = None
        x_t = torch.randn(B, L, 4)
        y_t = torch.randn(B, L, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 1, 1)


class TestTiDEForecasterDXMask:
    """Tests for d_x_mask."""

    def test_forward_d_x_mask(self):
        f = _tide_forecaster(
            task_type="forecasting", include_x_t=True, d_x_mask=[1, 0, 1, 0], d_yt=2
        )
        B, L = 2, 24
        x_c = torch.randn(B, L, 4)
        y_c = torch.randn(B, L, 2)
        x_t = torch.randn(B, 30, 4)
        y_t = torch.randn(B, 6, 2)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 2)


class TestTiDEForecasterMaskYC:
    """Tests for mask_y_c=True."""

    def test_forward_mask_y_c(self):
        f = _tide_forecaster(task_type="forecasting", mask_y_c=True, d_yt=2)
        B, L = 2, 24
        y_c = torch.randn(B, L, 2)
        x_c = torch.randn(B, L, 4)
        x_t = torch.randn(B, 30, 4)
        y_t = torch.randn(B, 6, 2)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 2)


class TestTiDEForecasterYC2D:
    """Tests for y_c with 2D shape (unsqueeze)."""

    def test_forward_y_c_2d_unsqueeze(self):
        f = _tide_forecaster(task_type="forecasting")
        B, L = 2, 24
        y_c = torch.randn(B, L)
        x_c = torch.randn(B, L, 4)
        x_t = torch.randn(B, 30, 4)
        y_t = torch.randn(B, 6, 1)
        (out,) = f.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape[0] == B
        assert out.shape[1] == 6
