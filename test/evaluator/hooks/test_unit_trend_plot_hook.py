"""
Tests for picid.evaluator.hooks.unit_trend_plot.UnitTrendPlotHook.

Ref: docs/evaluators/index.md - MultiUnitEvaluator, pred-vs-target per unit.
Validates: Guard conditions, mode/epoch filtering, subsampling, 1D/2D unit IDs.
"""

import numpy as np
from unittest.mock import MagicMock, patch

from picid.evaluator.hooks.unit_trend_plot import UnitTrendPlotHook


# =============================================================================
# === Tests: Guard Conditions ===
# =============================================================================


def test_unit_trend_plot_hook_returns_early_when_no_remote_logger():
    """
    Validates guard: hook exits when evaluator.remote_logger is None.

    Methodology: Mock evaluator with remote_logger=None; call on_compute_end.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - remote_logger for plot logging.
    """
    hook = UnitTrendPlotHook()
    evaluator = MagicMock()
    evaluator.remote_logger = None
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((5, 1, 1)),
        "targets": np.ones((5, 1, 1)),
        "unit_ids": np.array([1, 1, 2, 2, 2]),
    }

    with patch("picid.evaluator.hooks.unit_trend_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_plt.subplots.assert_not_called()


def test_unit_trend_plot_hook_returns_early_when_mode_train():
    """
    Validates guard: hook exits when mode == "train".

    Methodology: Call on_compute_end with mode="train".
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - log_image_every_n_epochs for val/test.
    """
    hook = UnitTrendPlotHook()
    evaluator = MagicMock()
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((5, 1, 1)),
        "targets": np.ones((5, 1, 1)),
        "unit_ids": np.array([1, 1, 2, 2, 2]),
    }

    with patch("picid.evaluator.hooks.unit_trend_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, mode="train", epoch=0, step=0)
        mock_plt.subplots.assert_not_called()


def test_unit_trend_plot_hook_returns_early_when_val_epoch_not_divisible():
    """
    Validates guard: hook exits when mode=="val" and epoch % log_every_n_epochs != 0.

    Methodology: Call with mode="val", epoch=1, log_every_n_epochs=10.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - log_image_every_n_epochs.
    """
    hook = UnitTrendPlotHook(log_every_n_epochs=10)
    evaluator = MagicMock()
    evaluator.remote_logger = MagicMock()
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((5, 1, 1)),
        "targets": np.ones((5, 1, 1)),
        "unit_ids": np.array([1, 1, 2, 2, 2]),
    }

    with patch("picid.evaluator.hooks.unit_trend_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, mode="val", epoch=1, step=0)
        mock_plt.subplots.assert_not_called()


def test_unit_trend_plot_hook_returns_early_when_no_unit_ids():
    """
    Validates guard: hook exits when unit_ids not in data.

    Methodology: buffer.get_all returns preds/targets without unit_ids.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - unit_id required for MultiUnitEvaluator.
    """
    hook = UnitTrendPlotHook()
    evaluator = MagicMock()
    evaluator.remote_logger = MagicMock()
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {
        "preds": np.ones((5, 1, 1)),
        "targets": np.ones((5, 1, 1)),
    }

    with patch("picid.evaluator.hooks.unit_trend_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, mode="val", epoch=0, step=0)
        mock_plt.subplots.assert_not_called()


# =============================================================================
# === Tests: Main Flow - Val Mode, Epoch 0 ===
# =============================================================================


@patch("picid.evaluator.hooks.unit_trend_plot.plt")
def test_unit_trend_plot_hook_plots_per_unit_1d_ids(
    mock_plt, phm_multiunit_buffer_data
):
    """
    Validates main flow: one plot per unique unit with 1D unit_ids.

    Methodology: phm_multiunit_buffer_data has unit_ids (15,) - 3 units.
    Gold Standard: Multi-unit degradation structure from conftest.
    Expected outcome: 3 figures (one per unit), log_plot called 3 times.
    Ref: docs/evaluators/index.md - pred-vs-target per unit.
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    hook = UnitTrendPlotHook()
    evaluator = MagicMock()
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()

    data = phm_multiunit_buffer_data
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = data

    hook.on_compute_end({}, evaluator, mode="val", epoch=0, step=0)

    assert mock_plt.subplots.call_count == 3
    assert evaluator.log_plot.call_count == 3


@patch("picid.evaluator.hooks.unit_trend_plot.plt")
def test_unit_trend_plot_hook_plots_per_unit_2d_ids(mock_plt):
    """
    Validates 2D unit_ids: np.unique(axis=0) and mask with .all(axis=1).

    Methodology: unit_ids shape (12, 2) - composite (dataset_id, unit_id).
    Expected outcome: One plot per unique row; plots created.
    Ref: docs/evaluators/index.md - unit_id can be 1D or 2D.
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    hook = UnitTrendPlotHook()
    evaluator = MagicMock()
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()

    unit_ids = np.array([[1, 1], [1, 1], [1, 2], [1, 2], [2, 1], [2, 1]])
    data = {
        "preds": np.random.rand(6, 1, 1).astype(np.float32),
        "targets": np.random.rand(6, 1, 1).astype(np.float32),
        "unit_ids": unit_ids,
    }
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = data

    hook.on_compute_end({}, evaluator, mode="val", epoch=0, step=0)

    assert mock_plt.subplots.call_count == 3
    evaluator.log_plot.assert_any_call(mock_fig, "unit_1_1_trend", "val", 0, 0)
    evaluator.log_plot.assert_any_call(mock_fig, "unit_1_2_trend", "val", 0, 0)
    evaluator.log_plot.assert_any_call(mock_fig, "unit_2_1_trend", "val", 0, 0)


@patch("picid.evaluator.hooks.unit_trend_plot.plt")
def test_unit_trend_plot_hook_subsamples_when_above_threshold(mock_plt):
    """
    Validates subsampling: when p.size > subsample_threshold, use step.

    Methodology: Unit with 2500 samples; subsample_threshold=2000, factor=10.
    Expected outcome: p and t sliced [::10] before plotting.
    Ref: docs/evaluators/index.md - plot_subsample_threshold, plot_subsample_factor.
    """
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    hook = UnitTrendPlotHook(
        subsample_threshold=2000,
        subsample_factor=10,
        enable_subsampling=True,
    )
    evaluator = MagicMock()
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()

    n_samples = 2500
    unit_ids = np.ones(n_samples, dtype=int)
    data = {
        "preds": np.random.rand(n_samples, 1, 1).astype(np.float32),
        "targets": np.random.rand(n_samples, 1, 1).astype(np.float32),
        "unit_ids": unit_ids,
    }
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = data

    hook.on_compute_end({}, evaluator, mode="val", epoch=0, step=0)

    mock_ax.plot.assert_called()
    call_args = mock_ax.plot.call_args_list[0][0]
    arr = call_args[0]
    assert len(arr) == 250
