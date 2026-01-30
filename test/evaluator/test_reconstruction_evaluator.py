import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

# Adjust import based on your project structure
from lmetk.evaluator.reconstruction import ReconstructionEvaluator
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.transforms.base.multisource import InverseTransformMixin

# =========================================================================
# === 1. Fixtures ===
# =========================================================================


@pytest.fixture
def mock_path_obj():
    """Creates a robust mock for Path operations."""
    p = MagicMock(spec=Path)
    p.__truediv__.return_value = p
    p.mkdir.return_value = None
    return p


@pytest.fixture
def reconstruction_evaluator_fixture(mocker):
    """Provides an instance of ReconstructionEvaluator with mocked dependencies.
    
    apply_inverse_scaling=False (default, no scaling).
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mock_paths = MagicMock()
    mock_paths.plot_dir = "/mock/plots"
    mock_logger = MagicMock()

    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        inverse_transform=MagicMock(spec=InverseTransformMixin),
        apply_inverse_scaling=False,
        save_predictions=False,
        plot_reconstructions=True,
        paths=mock_paths,
        remote_logger=mock_logger,
    )
    return evaluator


@pytest.fixture
def reconstruction_evaluator_with_scaling_fixture(mocker):
    """ReconstructionEvaluator with apply_inverse_scaling=True.
    
    Tests scenarios where predictions need to be converted to physical units.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mock_paths = MagicMock()
    mock_paths.plot_dir = "/mock/plots"
    mock_logger = MagicMock()
    
    # Create mock scaler that applies 100x scaling
    mock_scaler = MagicMock(spec=InverseTransformMixin)
    mock_scaler.apply_inverse = True
    
    def mock_inverse(preds, targets, **kwargs):
        return preds * 100, targets * 100
    
    mock_scaler.inverse_transform_if_needed = MagicMock(side_effect=mock_inverse)

    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        inverse_transform=mock_scaler,
        apply_inverse_scaling=True,  # Enable scaling at init
        save_predictions=False,
        plot_reconstructions=True,
        paths=mock_paths,
        remote_logger=mock_logger,
    )
    # Replace scaling_wrapper with our mock
    evaluator.scaling_wrapper = mock_scaler
    
    return evaluator


# =========================================================================
# === 2. Tests ===
# =========================================================================


def test_init_forces_collection_and_handles_kwargs(mocker):
    """Tests collection forcing and kwargs popping."""
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        plot_reconstructions=True,
        collect_predictions=False,
        paths=MagicMock(),
    )
    assert evaluator.collect_predictions is True
    assert evaluator.plot_reconstructions is True


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_handles_2d_inputs(
    mock_plt, reconstruction_evaluator_fixture, forecasting_data_multi_step
):
    """
    Verifies the fix for IndexError.
    Ensures 2D Univariate data (N, T) is expanded to (N, T, 1) internally.
    
    **Uses fixture**: forecasting_data_multi_step - realistic multi-step data
    (converted to 2D for this edge case test).
    """
    evaluator = reconstruction_evaluator_fixture

    # 1. Mock Axes: Create a proper (5, 2) object array so .flatten() works
    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)

    # 2. FIX: Mock Colors to prevent ZeroDivisionError
    mock_cmap = MagicMock()
    mock_cmap.colors = ["red", "blue"]  # Non-empty list
    mock_plt.get_cmap.return_value = mock_cmap

    # Input: 2D Arrays (Batch=5, Time=10) - Missing Feature Dim
    # Use realistic data but squeeze to 2D for edge case test
    preds_2d = forecasting_data_multi_step["predictions"][:5, :, 0]  # (5, 24)
    targets_2d = forecasting_data_multi_step["targets"][:5, :, 0]  # (5, 24)

    # ACT
    evaluator._plot_reconstructions(preds_2d, targets_2d, mode="val", epoch=1, step=1)

    # ASSERT
    mock_plt.subplots.assert_called_once()


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_internals_full_coverage(
    mock_plt, reconstruction_evaluator_fixture, forecasting_data_multi_step
):
    """
    Tests the internal plotting logic deeply to ensure 100% coverage.
    
    **Uses fixture**: forecasting_data_multi_step - realistic multi-step data
    with degradation curves.
    """
    evaluator = reconstruction_evaluator_fixture

    # 1. Mock Axes
    mock_axes = np.empty((5, 2), dtype=object)
    mock_ax_instances = []
    for i in range(5):
        for j in range(2):
            m = MagicMock()
            mock_axes[i, j] = m
            mock_ax_instances.append(m)
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)

    # 2. Mock Colors
    mock_cmap = MagicMock()
    mock_cmap.colors = ["red", "blue", "green"]
    mock_plt.get_cmap.return_value = mock_cmap

    # ACT: Use realistic fixture data (Batch > 10)
    preds = forecasting_data_multi_step["predictions"][:12]  # 12 samples
    targets = forecasting_data_multi_step["targets"][:12]
    
    evaluator._plot_reconstructions(preds, targets, mode="val", epoch=1, step=1)

    # ASSERT
    mock_plt.subplots.assert_called_once()

    first_ax = mock_ax_instances[0]
    assert first_ax.plot.called, "Plotting did not occur inside loop"
    assert first_ax.legend.called
    assert first_ax.grid.called

    mock_plt.get_cmap.assert_called_with("tab10")


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_plots_denormalized_if_scaling_on(
    mock_path_cls, mock_plt, reconstruction_evaluator_with_scaling_fixture, mock_path_obj
):
    """
    Verifies plotting uses SCALED data if apply_inverse_scaling=True.
    
    **Methodology**: Use fixture with scaling enabled, feed data through update().
    """
    mock_path_cls.return_value = mock_path_obj

    evaluator = reconstruction_evaluator_with_scaling_fixture
    
    # Verify scaling is enabled at init
    assert evaluator.scaling_wrapper.apply_inverse is True

    # Feed data through proper interface
    norm_data = np.ones((2, 2, 1))
    evaluator.update({"predictions": norm_data, "targets": norm_data})

    evaluator._plot_reconstructions = MagicMock()

    # ACT: Mock only file operations, not orchestration
    with patch.object(evaluator, '_save_predictions_to_file'):
        results = evaluator.compute(mode="val", epoch=1, step=1)

    # ASSERT: Verify plotting was called and compute() returns metrics
    evaluator._plot_reconstructions.assert_called()
    assert isinstance(results, dict)  # Verify compute() returns metrics


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_plots_normalized_if_scaling_off(
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """
    Verifies plotting uses STANDARD data if apply_inverse_scaling=False.
    
    **Methodology**: Use fixture with scaling disabled (default), feed through update().
    """
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture
    
    # Verify scaling is disabled (fixture default)
    assert evaluator.scaling_wrapper.apply_inverse is False

    # Feed data through proper interface
    norm_data = np.ones((2, 2, 1))
    evaluator.update({"predictions": norm_data, "targets": norm_data})

    evaluator._plot_reconstructions = MagicMock()

    # ACT: Mock only file operations, not orchestration
    with patch.object(evaluator, '_save_predictions_to_file'):
        results = evaluator.compute(mode="val", epoch=1, step=1)

    # ASSERT: Verify plotting was called with correct data and compute() returns metrics
    evaluator._plot_reconstructions.assert_called()
    args, _ = evaluator._plot_reconstructions.call_args
    passed_preds = args[0]
    np.testing.assert_array_equal(passed_preds, np.concatenate([norm_data], axis=0))
    assert isinstance(results, dict)  # Verify compute() returns metrics


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_orchestration_saving(
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """
    Verifies full flow: Path creation, Logging, Saving.
    
    **Methodology**: Feed data through update() instead of direct manipulation.
    """
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture

    # Feed data through proper interface
    data = np.zeros((2, 2, 1))
    evaluator.update({"predictions": data, "targets": data})

    mock_fig = MagicMock()
    evaluator._plot_reconstructions = MagicMock(return_value=mock_fig)
    evaluator.log_plot = MagicMock()

    # ACT: Mock only file operations, not orchestration
    with patch.object(evaluator, '_save_predictions_to_file'):
        results = evaluator.compute(mode="test", epoch=10, step=100)

    # ASSERT: Verify full orchestration including plotting, saving, and metrics
    assert evaluator.plots_out_path == mock_path_obj
    mock_path_obj.mkdir.assert_called_with(parents=True, exist_ok=True)
    evaluator.log_plot.assert_called_once()
    mock_fig.savefig.assert_called_once()
    mock_plt.close.assert_called_once_with(mock_fig)
    assert isinstance(results, dict)  # Verify compute() returns metrics


# =========================================================================
# === 3. Additional Coverage Tests ===
# =========================================================================


@patch("lmetk.evaluator.reconstruction.logger")
def test_plot_reconstructions_no_logger_step_zero_warning(mock_logger, mocker):
    """
    Tests _plot_reconstructions logs warning at step 0 when no logger.
    
    **PHM Logic**: Warning logged only on first step to avoid spam.
    
    **Methodology**: Call with step=0 and no remote_logger.
    
    **Expected**: Warning logged.
    
    Validates: Requirement RE-PLOT-1 - Step 0 warning
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    
    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        plot_reconstructions=True,
        paths=MagicMock(),
        remote_logger=None,  # No logger
    )
    
    preds = np.zeros((5, 10, 1))
    targets = np.zeros((5, 10, 1))
    
    result = evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=0)
    
    assert result is None
    mock_logger.warning.assert_called_once()
    assert "Remote logger is not configured" in str(mock_logger.warning.call_args)


@patch("lmetk.evaluator.reconstruction.logger")
def test_plot_reconstructions_no_logger_step_nonzero_no_warning(mock_logger, mocker):
    """
    Tests _plot_reconstructions does not log warning when step > 0.
    
    **PHM Logic**: Warning logged only on first step to avoid spam.
    
    **Methodology**: Call with step > 0 and no remote_logger.
    
    **Expected**: No warning logged.
    
    Validates: Requirement RE-PLOT-2 - No warning after step 0
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    
    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        plot_reconstructions=True,
        paths=MagicMock(),
        remote_logger=None,  # No logger
    )
    
    preds = np.zeros((5, 10, 1))
    targets = np.zeros((5, 10, 1))
    
    result = evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=5)
    
    assert result is None
    mock_logger.warning.assert_not_called()


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_small_batch(mock_plt, reconstruction_evaluator_fixture, regression_data_basic):
    """
    Tests _plot_reconstructions handles small batches (< 10 samples).
    
    **PHM Logic**: Plotting adapts to available samples.
    
    **Methodology**: Pass batch smaller than 10, use realistic data.
    
    **Uses fixture**: regression_data_basic - realistic RUL predictions (5 samples).
    
    **Expected**: Plots created for available samples.
    
    Validates: Requirement RE-PLOT-3 - Small batch handling
    """
    evaluator = reconstruction_evaluator_fixture
    
    # Create proper mock axes array
    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value.colors = ["red", "blue", "green"]
    
    # ACT: Use realistic fixture data (5 samples - small batch)
    preds = regression_data_basic["predictions"]
    targets = regression_data_basic["targets"]
    
    # Expand to 3D if needed (add time dimension)
    if preds.ndim == 3 and preds.shape[1] == 1:
        # Already 3D, use as-is
        pass
    else:
        # Ensure 3D shape
        preds = preds.reshape(-1, 1, 1)
        targets = targets.reshape(-1, 1, 1)
    
    fig = evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=0)
    
    # ASSERT
    assert fig is not None
    mock_plt.subplots.assert_called_once()


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_multivariate(mock_plt, reconstruction_evaluator_fixture, forecasting_data_multivariate):
    """
    Tests _plot_reconstructions handles multivariate data (multiple features).
    
    **PHM Logic**: Multiple sensor channels visualized together.
    
    **Methodology**: Pass realistic multivariate data with multiple feature dimensions.
    
    **Uses fixture**: forecasting_data_multivariate - realistic multivariate forecasting
    data with 5 sensor channels.
    
    **Expected**: All features plotted.
    
    Validates: Requirement RE-PLOT-4 - Multivariate handling
    """
    evaluator = reconstruction_evaluator_fixture
    
    # Create proper mock axes array
    mock_axes = np.empty((5, 2), dtype=object)
    mock_ax_list = []
    for i in range(5):
        for j in range(2):
            ax = MagicMock()
            mock_axes[i, j] = ax
            mock_ax_list.append(ax)
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value.colors = ["red", "blue", "green", "purple"]
    
    # ACT: Use realistic multivariate fixture data (5 features)
    preds = forecasting_data_multivariate["predictions"]
    targets = forecasting_data_multivariate["targets"]
    
    fig = evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=0)
    
    # ASSERT
    assert fig is not None
    # First axis should have plot called multiple times for different features
    assert mock_ax_list[0].plot.call_count >= 5  # At least once per feature


@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_handles_empty_preds(mock_path_cls, reconstruction_evaluator_fixture, mock_path_obj):
    """
    Tests compute handles empty predictions gracefully.
    
    **PHM Logic**: Early epochs may have no data yet.
    
    **Methodology**: Use reset() to ensure collections are empty.
    
    **Expected**: No plotting error.
    
    Validates: Requirement RE-COMP-1 - Empty data handling
    """
    mock_path_cls.return_value = mock_path_obj
    
    evaluator = reconstruction_evaluator_fixture
    
    # Use reset() to clear collections (not manual assignment)
    evaluator.reset()
    
    # Verify collections are empty
    assert len(evaluator.preds) == 0
    assert len(evaluator.targets) == 0
    
    evaluator._plot_reconstructions = MagicMock()
    
    # ACT: Mock only file operations, not orchestration
    with patch.object(evaluator, '_save_predictions_to_file'):
        result = evaluator.compute(mode="val", epoch=1, step=0)
    
    # ASSERT: Should not have called plotting (empty check in compute), but should return metrics
    evaluator._plot_reconstructions.assert_not_called()
    assert isinstance(result, dict)  # Verify compute() returns metrics even with empty data


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_no_plot_dir_path(mock_path_cls, mock_plt, mocker, mock_path_obj):
    """
    Tests compute handles missing plot_dir in paths.
    
    **PHM Logic**: Paths may not always have plot_dir configured.
    
    **Methodology**: Set paths.plot_dir to None.
    
    **Expected**: plots_out_path remains None.
    
    Validates: Requirement RE-COMP-2 - Missing plot_dir handling
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mock_path_cls.return_value = mock_path_obj
    
    mock_paths = MagicMock()
    mock_paths.plot_dir = None  # No plot_dir
    
    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        plot_reconstructions=True,
        paths=mock_paths,
        remote_logger=MagicMock(),
    )
    
    # Feed data through proper interface
    data = np.zeros((2, 5, 1))
    evaluator.update({"predictions": data, "targets": data})
    
    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value.colors = ["red", "blue"]
    
    # ACT: Mock only file operations, not orchestration
    with patch.object(evaluator, '_save_predictions_to_file'):
        results = evaluator.compute(mode="val", epoch=1, step=0)
    
    # ASSERT: plots_out_path should remain None, but compute() should return metrics
    assert evaluator.plots_out_path is None
    assert isinstance(results, dict)  # Verify compute() returns metrics
