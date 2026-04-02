"""Tests for :class:`picid.metrics.manager.MetricManager` (forecasting + dual scaling)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from picid.metrics.manager import MetricManager


@pytest.fixture
def mock_metric():
    m = MagicMock()
    m.compute.return_value = 0.42
    return m


def test_forecasting_task_uses_regression_style_metric_factory(mocker, mock_metric):
    """Forecasting builds metrics like regression (not classification)."""
    create = mocker.patch(
        "picid.metrics.manager.MetricFactory.create_metric", return_value=mock_metric
    )
    mm = MetricManager(metric_names=["mse"], task_type="forecasting", paths={})
    assert "mse" in mm.metrics
    create.assert_called_once_with("mse", {})


def test_forecasting_compute_single_suffix_when_not_dual(mock_metric, mocker):
    mocker.patch(
        "picid.metrics.manager.MetricFactory.create_metric", return_value=mock_metric
    )
    mm = MetricManager(metric_names=["mse"], task_type="forecasting", is_dual=False)
    preds = np.zeros((2, 3, 4))
    targets = np.zeros_like(preds)
    mm.update(preds, targets)
    out = mm.compute()
    assert out == {"mse_normalized": 0.42}


def test_forecasting_dual_scaling_updates_both_branches(mock_metric, mocker):
    mocker.patch(
        "picid.metrics.manager.MetricFactory.create_metric", return_value=mock_metric
    )
    mm = MetricManager(metric_names=["mse"], task_type="forecasting", is_dual=True)
    preds = np.ones((1, 2, 3))
    targets = np.ones_like(preds)
    norm_p = np.zeros_like(preds)
    norm_t = np.zeros_like(targets)
    mm.update(preds, targets, norm_preds=norm_p, norm_targets=norm_t)
    out = mm.compute()
    assert out["mse_denormalized"] == 0.42
    assert out["mse_normalized"] == 0.42
