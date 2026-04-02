import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from picid.evaluator.reconstruction import ReconstructionEvaluator
from picid.evaluator.hooks.reconstruction_plot import ReconstructionPlotHook
from picid.metrics.metric_factory import MetricFactory

# =========================================================================
# === 1. Fixtures ===
# =========================================================================


@pytest.fixture
def base_evaluator_kwargs(mocker):
    """Common setup for evaluator initialization."""
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    # Flexible mock for the scaler that won't throw AttributeErrors
    mock_scaler = MagicMock()
    mock_scaler.apply_inverse = False
    mock_scaler.inverse_transform_if_needed.side_effect = lambda p, t, **k: (p, t)

    return {
        "metric_names": ["mse"],
        "paths": MagicMock(plot_dir="/mock/plots", eval_details="/mock/eval"),
        "remote_logger": MagicMock(),
        "scaling_wrapper": mock_scaler,
    }


@pytest.fixture
def reconstruction_evaluator_fixture(base_evaluator_kwargs):
    """Standard evaluator with the Plot Hook enabled."""
    evaluator = ReconstructionEvaluator(
        hooks=[ReconstructionPlotHook()], **base_evaluator_kwargs
    )
    # Essential: Hook guards check these attributes
    evaluator.plot_reconstructions = True
    evaluator.scaling_wrapper = base_evaluator_kwargs["scaling_wrapper"]
    return evaluator


# =========================================================================
# === 2. Tests ===
# =========================================================================


def test_init_smart_activation(mocker):
    """Verify smart activation: collect_predictions logic."""
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    # Hooks present -> Auto-enable collection
    eval_auto = ReconstructionEvaluator(
        metric_names=["mse"], hooks=[ReconstructionPlotHook()]
    )
    assert eval_auto.collect_predictions is True

    # Explicit override -> Respect user choice
    eval_manual = ReconstructionEvaluator(
        metric_names=["mse"],
        hooks=[ReconstructionPlotHook()],
        collect_predictions=False,
    )
    assert eval_manual.collect_predictions is False


@patch("picid.evaluator.hooks.reconstruction_plot.plt")
def test_validation_enforces_phm_shape(mock_plt, reconstruction_evaluator_fixture):
    """Verifies that 2D inputs fail validation before PHM channel-width checks."""
    evaluator = reconstruction_evaluator_fixture

    # 2D data fails the (N, T, C) requirement first
    bad_data = {"predictions": np.zeros((5, 10)), "targets": np.zeros((5, 10))}

    with pytest.raises(AssertionError, match="Targets must be 3D"):
        evaluator.update(bad_data)


@patch("picid.evaluator.hooks.reconstruction_plot.plt")
def test_plot_reconstructions_flow(mock_plt, reconstruction_evaluator_fixture):
    """Verifies orchestration from update through plotting."""
    # FIX: Provide return values for unpacking (fig, axes)
    mock_plt.subplots.return_value = (
        MagicMock(),
        MagicMock(flatten=MagicMock(return_value=[MagicMock()] * 10)),
    )

    evaluator = reconstruction_evaluator_fixture
    data = np.random.rand(5, 10, 1)  # Correct PHM Shape
    evaluator.update({"predictions": data, "targets": data})

    evaluator.compute(mode="val", epoch=1, step=1)

    mock_plt.subplots.assert_called()
    mock_plt.close.assert_called()


@patch("picid.evaluator.hooks.reconstruction_plot.plt")
def test_dual_scaling_logic(mock_plt, reconstruction_evaluator_fixture):
    """Verifies that scaling is applied during the update process."""
    mock_plt.subplots.return_value = (
        MagicMock(),
        MagicMock(flatten=MagicMock(return_value=[MagicMock()] * 10)),
    )

    evaluator = reconstruction_evaluator_fixture
    evaluator.scaling_wrapper.apply_inverse = True
    evaluator.scaling_wrapper.inverse_transform_if_needed.side_effect = (
        lambda p, t, **k: (p * 10, t * 10)
    )

    data = np.ones((2, 5, 1))
    evaluator.update({"predictions": data, "targets": data})

    # Verify the scaler was used
    all_data = evaluator.buffer.get_all()
    assert np.all(all_data["preds"] == 10.0)

    evaluator.compute(mode="val", epoch=1, step=1)
    mock_plt.subplots.assert_called()


def test_compute_handles_empty_buffer(reconstruction_evaluator_fixture):
    """Ensures hook exits early if no predictions were collected."""
    evaluator = reconstruction_evaluator_fixture
    evaluator.reset()

    with patch("picid.evaluator.hooks.reconstruction_plot.plt.subplots") as mock_sub:
        evaluator.compute(mode="val", epoch=1, step=0)
        mock_sub.assert_not_called()


def test_plot_reconstructions_returns_early_without_logger(
    reconstruction_evaluator_fixture,
):
    """Verifies the hook returns early if remote_logger is missing."""
    evaluator = reconstruction_evaluator_fixture
    evaluator.remote_logger = None  # Remove logger

    data = np.zeros((2, 5, 1))
    evaluator.update({"predictions": data, "targets": data})

    with patch("picid.evaluator.hooks.reconstruction_plot.plt.subplots") as mock_plt:
        evaluator.compute(mode="val", epoch=1, step=0)
        # It should return early before calling subplots
        mock_plt.assert_not_called()
