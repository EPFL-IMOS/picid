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
    """Provides an instance of ReconstructionEvaluator with mocked dependencies."""
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
    mock_plt, reconstruction_evaluator_fixture
):
    """
    Verifies the fix for IndexError.
    Ensures 2D Univariate data (N, T) is expanded to (N, T, 1) internally.
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
    preds_2d = np.zeros((5, 10))
    targets_2d = np.zeros((5, 10))

    # ACT
    evaluator._plot_reconstructions(preds_2d, targets_2d, mode="val", epoch=1, step=1)

    # ASSERT
    mock_plt.subplots.assert_called_once()


@patch("lmetk.evaluator.reconstruction.plt")
def test_plot_reconstructions_internals_full_coverage(
    mock_plt, reconstruction_evaluator_fixture
):
    """
    Tests the internal plotting logic deeply to ensure 100% coverage.
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

    # Setup Data (Batch > 10)
    preds = np.zeros((12, 10, 1))
    targets = np.zeros((12, 10, 1))

    # ACT
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
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """Verifies plotting uses SCALED data if apply_inverse_scaling=True."""
    mock_path_cls.return_value = mock_path_obj

    evaluator = reconstruction_evaluator_fixture
    evaluator.scaling_wrapper.apply_inverse = True

    norm_data = np.ones((2, 2, 1))
    denorm_data = np.ones((2, 2, 1)) * 100

    evaluator.preds.append(norm_data)
    evaluator.targets.append(norm_data)
    evaluator.scaled_preds.append(denorm_data)
    evaluator.scaled_targets.append(denorm_data)

    evaluator._plot_reconstructions = MagicMock()

    with patch("lmetk.evaluator.default.DefaultEvaluator.compute", return_value={}):
        evaluator.compute(mode="val", epoch=1, step=1)

        args, _ = evaluator._plot_reconstructions.call_args
        passed_preds = args[0]
        np.testing.assert_array_equal(
            passed_preds, np.concatenate([denorm_data], axis=0)
        )


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_plots_normalized_if_scaling_off(
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """Verifies plotting uses STANDARD data if apply_inverse_scaling=False."""
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture
    evaluator.scaling_wrapper.apply_inverse = False

    norm_data = np.ones((2, 2, 1))
    evaluator.preds.append(norm_data)
    evaluator.targets.append(norm_data)

    evaluator._plot_reconstructions = MagicMock()

    with patch("lmetk.evaluator.default.DefaultEvaluator.compute", return_value={}):
        evaluator.compute(mode="val", epoch=1, step=1)

        args, _ = evaluator._plot_reconstructions.call_args
        passed_preds = args[0]
        np.testing.assert_array_equal(passed_preds, np.concatenate([norm_data], axis=0))


@patch("lmetk.evaluator.reconstruction.plt")
@patch("lmetk.evaluator.reconstruction.Path")
def test_compute_orchestration_saving(
    mock_path_cls, mock_plt, reconstruction_evaluator_fixture, mock_path_obj
):
    """Verifies full flow: Path creation, Logging, Saving."""
    mock_path_cls.return_value = mock_path_obj
    evaluator = reconstruction_evaluator_fixture

    data = np.zeros((2, 2, 1))
    evaluator.preds.append(data)
    evaluator.targets.append(data)

    mock_fig = MagicMock()
    evaluator._plot_reconstructions = MagicMock(return_value=mock_fig)
    evaluator.log_plot = MagicMock()

    with patch("lmetk.evaluator.default.DefaultEvaluator.compute", return_value={}):
        evaluator.compute(mode="test", epoch=10, step=100)

        assert evaluator.plots_out_path == mock_path_obj
        mock_path_obj.mkdir.assert_called_with(parents=True, exist_ok=True)

        evaluator.log_plot.assert_called_once()
        mock_fig.savefig.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)
