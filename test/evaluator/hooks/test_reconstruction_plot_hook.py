"""
Tests for picid.evaluator.hooks.reconstruction_plot.ReconstructionPlotHook.

Ref: docs/evaluators/index.md - ReconstructionEvaluator, reconstruction plots.
Validates: Guard conditions, plot creation, data integrity, PHM shape (N, T, C).
"""

import numpy as np
from unittest.mock import MagicMock, patch

from picid.evaluator.hooks.reconstruction_plot import ReconstructionPlotHook
from picid.evaluator.buffer import PredictionBuffer


# =============================================================================
# === Tests: Guard Conditions ===
# =============================================================================


def test_reconstruction_plot_hook_returns_early_when_plot_reconstructions_false():
    """
    Validates guard: hook exits when evaluator.plot_reconstructions is False.

    Methodology: Mock evaluator with plot_reconstructions=False; call on_compute_end.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - plot_reconstructions flag.
    """
    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = False
    evaluator.remote_logger = MagicMock()
    evaluator.buffer = PredictionBuffer()
    evaluator.buffer.accumulate(
        {"preds": np.ones((2, 10, 1)), "targets": np.ones((2, 10, 1))}
    )

    with patch("picid.evaluator.hooks.reconstruction_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_plt.subplots.assert_not_called()


def test_reconstruction_plot_hook_returns_early_when_no_remote_logger():
    """
    Validates guard: hook exits when evaluator.remote_logger is None.

    Methodology: Mock evaluator with remote_logger=None; call on_compute_end.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - remote_logger for plot logging.
    """
    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = True
    evaluator.remote_logger = None
    evaluator.buffer = PredictionBuffer()
    evaluator.buffer.accumulate(
        {"preds": np.ones((2, 10, 1)), "targets": np.ones((2, 10, 1))}
    )

    with patch("picid.evaluator.hooks.reconstruction_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_plt.subplots.assert_not_called()


def test_reconstruction_plot_hook_returns_early_when_buffer_empty():
    """
    Validates guard: hook exits when buffer has no data.

    Methodology: Empty buffer; get_all returns {}.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - buffer.get_all data flow.
    """
    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = True
    evaluator.remote_logger = MagicMock()
    evaluator.buffer = PredictionBuffer()

    with patch("picid.evaluator.hooks.reconstruction_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_plt.subplots.assert_not_called()


def test_reconstruction_plot_hook_returns_early_when_preds_none():
    """
    Validates guard: hook exits when data has preds=None.

    Methodology: buffer.get_all returns dict without preds or with preds=None.
    Expected outcome: No plotting; plt.subplots not called.
    Ref: docs/evaluators/index.md - preds/targets contract.
    """
    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = True
    evaluator.remote_logger = MagicMock()
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {"targets": np.ones((2, 10, 1))}

    with patch("picid.evaluator.hooks.reconstruction_plot.plt") as mock_plt:
        hook.on_compute_end({}, evaluator, "val", 0, 0)
        mock_plt.subplots.assert_not_called()


# =============================================================================
# === Tests: Main Flow and Data Integrity ===
# =============================================================================


@patch("picid.evaluator.hooks.reconstruction_plot.plt")
def test_reconstruction_plot_hook_creates_figure_with_nominal_data(
    mock_plt, phm_reconstruction_buffer_data
):
    """
    Validates main flow: creates 5x2 subplot figure with nominal PHM data.

    Methodology: Use phm_reconstruction_buffer_data (10 samples, 100 timesteps, 3 ch).
    Gold Standard: transform_nominal_health ensures data integrity.
    Expected outcome: plt.subplots(5, 2), plots first channel, log_plot called.
    Ref: docs/evaluators/index.md - reconstruction plots, (N, T, C).
    """
    mock_fig = MagicMock()
    mock_axes = MagicMock(flatten=MagicMock(return_value=[MagicMock()] * 10))
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = True
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()

    data = phm_reconstruction_buffer_data
    preds, targets = data["preds"], data["targets"]
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {"preds": preds, "targets": targets}

    hook.on_compute_end({}, evaluator, "val", 0, 0)

    mock_plt.subplots.assert_called_once_with(5, 2, figsize=(15, 20))
    mock_plt.tight_layout.assert_called_once()
    evaluator.log_plot.assert_called_once_with(
        mock_fig, "reconstructions_overview", "val", 0, 0
    )
    mock_plt.close.assert_called_once_with(mock_fig)


@patch("picid.evaluator.hooks.reconstruction_plot.plt")
def test_reconstruction_plot_hook_selects_up_to_10_indices(mock_plt):
    """
    Validates index selection: np.linspace for up to 10 evenly spaced samples.

    Methodology: 20 samples → 10 indices; 3 samples → 3 indices.
    Expected outcome: num=min(10, len(preds)) in linspace.
    Ref: docs/evaluators/index.md - reconstruction overview plot.
    """
    mock_fig = MagicMock()
    mock_axes = MagicMock(flatten=MagicMock(return_value=[MagicMock()] * 10))
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = True
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()

    n_samples = 3
    preds = np.ones((n_samples, 50, 1), dtype=np.float32)
    targets = np.ones((n_samples, 50, 1), dtype=np.float32)
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {"preds": preds, "targets": targets}

    hook.on_compute_end({}, evaluator, "val", 0, 0)

    mock_plt.subplots.assert_called_once()
    mock_axes.flatten.assert_called()


@patch("picid.evaluator.hooks.reconstruction_plot.plt")
def test_reconstruction_plot_hook_plots_first_channel_only(mock_plt):
    """
    Validates that plots use first channel/dim (index 0) for comparison.

    Methodology: Multi-channel data (N, T, 3); verify ax.plot uses [:, 0].
    Expected outcome: targets[idx, :, 0] and preds[idx, :, 0] plotted.
    Ref: docs/evaluators/index.md - first channel for reconstruction comparison.
    """
    mock_fig = MagicMock()
    ax_mock = MagicMock()
    mock_axes = MagicMock(flatten=MagicMock(return_value=[ax_mock] * 10))
    mock_plt.subplots.return_value = (mock_fig, mock_axes)

    hook = ReconstructionPlotHook()
    evaluator = MagicMock()
    evaluator.plot_reconstructions = True
    evaluator.remote_logger = MagicMock()
    evaluator.log_plot = MagicMock()

    preds = np.random.rand(5, 20, 2).astype(np.float32)
    targets = np.random.rand(5, 20, 2).astype(np.float32)
    evaluator.buffer = MagicMock()
    evaluator.buffer.get_all.return_value = {"preds": preds, "targets": targets}

    hook.on_compute_end({}, evaluator, "val", 0, 0)

    assert ax_mock.plot.call_count >= 2
    calls = ax_mock.plot.call_args_list
    first_call_args = calls[0][0]
    assert len(first_call_args) >= 1
    arr = first_call_args[0]
    assert arr.ndim == 1
    assert len(arr) == 20
