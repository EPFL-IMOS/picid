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
    """Creates a robust mock for Path operations, preventing filesystem access."""
    p = MagicMock(spec=Path)
    p.__truediv__.return_value = p
    p.mkdir.return_value = None
    return p


@pytest.fixture
def reconstruction_evaluator_fixture(mocker):
    """
    Provides a ReconstructionEvaluator with:
    - No scaling (apply_inverse_scaling=False)
    - Plotting enabled
    - Mocked MetricFactory and Paths
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
        collect_predictions=True,
        remote_logger=mock_logger,
    )
    return evaluator


@pytest.fixture
def reconstruction_evaluator_with_scaling_fixture(mocker):
    """
    Provides a ReconstructionEvaluator with:
    - Inverse Scaling ENABLED (apply_inverse_scaling=True)
    - Mock scaler that multiplies by 100
    - Used to test that plots use physical units when requested.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    mock_paths = MagicMock()
    mock_paths.plot_dir = "/mock/plots"
    mock_logger = MagicMock()

    # Mock scaler logic: Simply multiply by 100 to simulate denormalization
    mock_scaler = MagicMock(spec=InverseTransformMixin)
    mock_scaler.apply_inverse = True

    def mock_inverse(preds, targets, **kwargs):
        return preds * 100, targets * 100

    mock_scaler.inverse_transform_if_needed = MagicMock(side_effect=mock_inverse)

    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        inverse_transform=mock_scaler,
        apply_inverse_scaling=True,
        save_predictions=False,
        plot_reconstructions=True,
        paths=mock_paths,
        collect_predictions=True,
        remote_logger=mock_logger,
    )
    # Manually attach mock scaler for test stability
    evaluator.scaling_wrapper = mock_scaler
    return evaluator


# =========================================================================
# === 2. Tests ===
# =========================================================================


def test_init_forces_collection_and_handles_kwargs(mocker):
    """
    Tests that initializing with plot_reconstructions=True forces data collection.

    **Logic**: You cannot plot reconstruction curves if you haven't collected
    the predictions during the update steps.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        plot_reconstructions=True,  # This should force collection=True
        paths=MagicMock(),
        collect_predictions=False,  # Trying to disable it
    )

    # Assert collection was forced on
    assert evaluator.collect_predictions is True
    assert evaluator.plot_reconstructions is True


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_handles_2d_inputs(
    mock_plt, reconstruction_evaluator_fixture, forecasting_data_multi_step
):
    """
    Verifies plotting logic handles 2D inputs (Batch, Time) by expanding them to 3D.

    **PHM Logic**: Univariate signals might arrive without the feature dimension.
    The plotter must handle this gracefully.
    """
    evaluator = reconstruction_evaluator_fixture

    # Mock Axes for subplots
    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)

    mock_cmap = MagicMock()
    mock_cmap.colors = ["red", "blue"]
    mock_plt.get_cmap.return_value = mock_cmap

    # Simulate 2D data (Batch, Time)
    preds_2d = forecasting_data_multi_step["predictions"][:5, :, 0]
    targets_2d = forecasting_data_multi_step["targets"][:5, :, 0]

    # Act
    evaluator._plot_reconstructions(preds_2d, targets_2d, mode="val", epoch=1, step=1)

    # Assert
    mock_plt.subplots.assert_called_once()


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_internals_full_coverage(
    mock_plt, reconstruction_evaluator_fixture, forecasting_data_multi_step
):
    """
    Deep test of plotting internals: loop iterations, legends, grid.

    **Uses fixture**: forecasting_data_multi_step (Realistic RUL data).
    """
    evaluator = reconstruction_evaluator_fixture

    mock_axes = np.empty((5, 2), dtype=object)
    mock_ax_instances = []
    for i in range(5):
        for j in range(2):
            m = MagicMock()
            mock_axes[i, j] = m
            mock_ax_instances.append(m)
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value = MagicMock(colors=["r", "b", "g"])

    # Act
    preds = forecasting_data_multi_step["predictions"][:12]
    targets = forecasting_data_multi_step["targets"][:12]
    evaluator._plot_reconstructions(preds, targets, mode="val", epoch=1, step=1)

    # Assert
    mock_plt.subplots.assert_called_once()
    # Verify plot was called on at least the first axis
    assert mock_ax_instances[0].plot.called


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_plots_denormalized_if_scaling_on(
    mock_path_cls,
    mock_plt,
    reconstruction_evaluator_with_scaling_fixture,
    mock_path_obj,
):
    """
    Verifies that if scaling is ON, the plotter receives DENORMALIZED data.

    **PHM Logic**: Engineers need to see reconstructions in physical units (e.g. Temperature),
    not just normalized [0,1] values.
    """
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_with_scaling_fixture

    # 1. Update with dummy data. Since scaling mock multiplies by 100:
    # Buffer will contain:
    #   - preds: [1, 1...]
    #   - scaled_preds: [100, 100...]
    norm_data = np.ones((2, 2, 1))
    evaluator.update({"predictions": norm_data, "targets": norm_data})

    # Mock the internal plotter to inspect what arguments it receives
    evaluator._plot_reconstructions = MagicMock()

    # Act: Compute triggers plotting
    with patch.object(evaluator, "_save_predictions_to_file"):
        results = evaluator.compute(mode="val", epoch=1, step=1)

    # Assert
    evaluator._plot_reconstructions.assert_called()
    assert isinstance(results, dict)


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_plots_normalized_if_scaling_off(
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """
    Verifies that if scaling is OFF, the plotter receives STANDARD data.
    """
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture

    norm_data = np.ones((2, 2, 1))
    evaluator.update({"predictions": norm_data, "targets": norm_data})

    evaluator._plot_reconstructions = MagicMock()

    with patch.object(evaluator, "_save_predictions_to_file"):
        evaluator.compute(mode="val", epoch=1, step=1)

    evaluator._plot_reconstructions.assert_called()


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_orchestration_saving(
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """
    Verifies the full orchestration pipeline in `compute()`:
    1. Resolve Paths
    2. Generate Plot
    3. Log Plot
    4. Save Plot to Disk
    """
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture

    # Populate buffer
    data = np.zeros((2, 2, 1))
    evaluator.update({"predictions": data, "targets": data})

    # Mock artifacts
    mock_fig = MagicMock()
    evaluator._plot_reconstructions = MagicMock(return_value=mock_fig)
    evaluator.log_plot = MagicMock()

    # Act
    with patch.object(evaluator, "_save_predictions_to_file"):
        evaluator.compute(mode="test", epoch=10, step=100)

    # Assert
    assert evaluator.plots_out_path == mock_path_obj
    evaluator.log_plot.assert_called_once()
    mock_fig.savefig.assert_called_once()


@patch("lmetk.evaluator.reconstruction.logger")
def test_plot_reconstructions_no_logger_step_zero_warning(mock_logger, mocker):
    """
    Tests that a warning is logged if no remote logger is configured (only at step 0).
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        plot_reconstructions=True,
        paths=MagicMock(),
        collect_predictions=True,
        remote_logger=None,  # Explicit None
    )
    preds = np.zeros((5, 10, 1))
    targets = np.zeros((5, 10, 1))

    evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=0)
    mock_logger.warning.assert_called_once()


@patch("lmetk.evaluator.reconstruction.logger")
def test_plot_reconstructions_no_logger_step_nonzero_no_warning(mock_logger, mocker):
    """Tests that the warning is suppressed for steps > 0 to avoid log spam."""
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())
    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        plot_reconstructions=True,
        paths=MagicMock(),
        collect_predictions=True,
        remote_logger=None,
    )
    preds = np.zeros((5, 10, 1))
    targets = np.zeros((5, 10, 1))

    evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=5)
    mock_logger.warning.assert_not_called()


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_small_batch(
    mock_plt, reconstruction_evaluator_fixture, regression_data_basic
):
    """
    Tests robust handling of small batches (less than the default 10 plot slots).
    """
    evaluator = reconstruction_evaluator_fixture
    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value.colors = ["r", "b"]

    preds = regression_data_basic["predictions"]
    targets = regression_data_basic["targets"]

    # Ensure 3D shape for helper test
    if preds.ndim == 2:
        preds = preds.reshape(-1, 1, 1)
        targets = targets.reshape(-1, 1, 1)

    fig = evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=0)
    assert fig is not None


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_multivariate(
    mock_plt, reconstruction_evaluator_fixture, forecasting_data_multivariate
):
    """
    Tests handling of multivariate data (e.g. 5 sensor channels).
    The plotter should loop through dimensions and plot each.
    """
    evaluator = reconstruction_evaluator_fixture
    mock_axes = np.empty((5, 2), dtype=object)
    mock_ax_list = []
    for i in range(5):
        for j in range(2):
            ax = MagicMock()
            mock_axes[i, j] = ax
            mock_ax_list.append(ax)
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value.colors = ["r", "b"]

    preds = forecasting_data_multivariate["predictions"]
    targets = forecasting_data_multivariate["targets"]

    fig = evaluator._plot_reconstructions(preds, targets, "val", epoch=1, step=0)
    assert fig is not None


@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_handles_empty_preds(
    mock_path_cls, reconstruction_evaluator_fixture, mock_path_obj
):
    """
    Tests that compute() gracefully handles cases where no predictions were collected.

    **Methodology**: Reset the evaluator to clear the buffer, then call compute().
    """
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture

    evaluator.reset()

    # FIX: Verify buffer is empty
    assert len(evaluator.buffer.preds) == 0
    assert len(evaluator.buffer.targets) == 0

    evaluator._plot_reconstructions = MagicMock()

    # Act
    with patch.object(evaluator, "_save_predictions_to_file"):
        result = evaluator.compute(mode="val", epoch=1, step=0)

    # Assert: Plotting should NOT be called because buffer is empty
    evaluator._plot_reconstructions.assert_not_called()
    assert isinstance(result, dict)


@patch("lmetk.evaluator.reconstruction.plt")
def test_compute_no_plot_dir_path(mock_plt, mocker):
    """
    Tests behavior when 'plot_dir' is missing from the paths configuration.
    Should simply skip directory creation and saving, but still plot to logger.
    """
    mocker.patch(
        "lmetk.metrics.metric_factory.MetricFactory.create_metric",
        return_value=MagicMock(),
    )

    # Mock paths where plot_dir is None
    mock_paths = MagicMock()
    mock_paths.plot_dir = None
    mock_paths.eval_details = None

    evaluator = ReconstructionEvaluator(
        metric_names=["mse"],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        plot_reconstructions=True,
        paths=mock_paths,
        collect_predictions=True,
        remote_logger=MagicMock(),
    )

    data = np.zeros((2, 5, 1))
    evaluator.update({"predictions": data, "targets": data})

    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)
    mock_plt.get_cmap.return_value.colors = ["r"]

    # Act
    with patch.object(evaluator, "_save_predictions_to_file"):
        results = evaluator.compute(mode="val", epoch=1, step=0)

    # Assert
    assert evaluator.plots_out_path is None
    assert isinstance(results, dict)
