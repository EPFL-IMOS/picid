import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from collections import defaultdict
from pathlib import Path

from lmetk.evaluator.multiunit import MultiUnitEvaluator
from lmetk.metrics.base import AbstractMetric

# =========================================================================
# === 1. MOCKS & FACTORY ===
# =========================================================================


@pytest.fixture
def mock_path_obj():
    """Robust path mock preventing filesystem access."""
    p = MagicMock(spec=Path)
    p.__truediv__.return_value = p
    p.mkdir.return_value = None
    return p


@pytest.fixture
def multiunit_evaluator_factory(mocker, mock_path_obj):
    """
    Factory fixture for creating MultiUnitEvaluator with custom configurations.

    This ensures that component-based evaluators are initialized with the
    correct mock dependencies.
    """

    def _create_evaluator(
        apply_inverse_scaling=False,
        log_image=True,
        log_per_unit_metrics=True,
        enable_plot_subsampling=False,
        plot_subsample_threshold=1000,
        plot_subsample_factor=10,
        remote_logger=None,
        save_predictions=True,
        collect_predictions=True,
        use_mock_logger=True,
    ):
        from lmetk.transforms.base.multisource import InverseTransformMixin

        class SimpleScaler(InverseTransformMixin):
            def __init__(self, apply_inverse_scaling):
                self.apply_inverse = apply_inverse_scaling

            def inverse_transform_if_needed(self, preds, targets, metadata=None):
                if self.apply_inverse:
                    return preds * 2.0, targets * 2.0
                return preds, targets

            def inverse_transform(self, data, metadata=None):
                """Inverse transforms a single data segment using the internal StandardScaler."""
                keys = list(data.keys())
                assert (
                    len(keys) == 1
                ), "StandardScaler only supports single key data_segment."

                np_data = data[keys[0]]

                if np_data.ndim == 1:
                    np_data = np_data.reshape(-1, 1)

                inverse_transformed_np_data = np_data * 2
                return inverse_transformed_np_data

        scaler = SimpleScaler(apply_inverse_scaling)

        if remote_logger is None and use_mock_logger:
            remote_logger = MagicMock()
            # Ensure DefaultEvaluator.log_plot can access .experiment.log
            remote_logger.experiment = MagicMock()

        return MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=scaler,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            collect_predictions=collect_predictions,
            paths=MagicMock(eval_details=mock_path_obj),
            remote_logger=remote_logger,
            log_per_unit_metrics=log_per_unit_metrics,
            log_image=log_image,
            enable_plot_subsampling=enable_plot_subsampling,
            plot_subsample_threshold=plot_subsample_threshold,
            plot_subsample_factor=plot_subsample_factor,
        )

    return _create_evaluator


# =========================================================================
# === FIXTURES ===
# =========================================================================


@pytest.fixture
def multiunit_evaluator_fixture(multiunit_evaluator_factory):
    """Basic MultiUnitEvaluator with scaling OFF and plotting ON."""
    return multiunit_evaluator_factory(apply_inverse_scaling=False, log_image=True)


@pytest.fixture
def multiunit_evaluator_with_scaling_fixture(multiunit_evaluator_factory):
    """MultiUnitEvaluator with scaling ON."""
    return multiunit_evaluator_factory(apply_inverse_scaling=True)


@pytest.fixture
def multiunit_evaluator_with_plotting_fixture(multiunit_evaluator_factory):
    """MultiUnitEvaluator specifically configured for plotting checks."""
    return multiunit_evaluator_factory(log_image=True, enable_plot_subsampling=False)


# =========================================================================
# === 2. TESTS ===
# =========================================================================


def test_init_creates_collections(multiunit_evaluator_fixture):
    """Tests that buffer and metric collections are initialized."""
    eval = multiunit_evaluator_fixture
    assert isinstance(eval.buffer.unit_ids, list)
    assert isinstance(eval.unit_metrics, defaultdict)


def test_reset_clears_everything(multiunit_evaluator_fixture):
    """Tests that reset clears buffer and per-unit metrics."""
    eval = multiunit_evaluator_fixture
    data = {
        "predictions": np.random.rand(5, 1, 1),
        "targets": np.random.rand(5, 1, 1),
        "unit_id": np.array([1, 1, 2, 2, 3]),
    }
    eval.update(data)
    assert len(eval.buffer.unit_ids) > 0

    eval.reset()
    assert len(eval.buffer.unit_ids) == 0
    assert len(eval.unit_metrics) == 0


def test_update_splits_data_by_unit_1d(multiunit_evaluator_fixture):
    """Tests correct routing of data to unit-specific metrics."""
    eval = multiunit_evaluator_fixture
    data = {
        "predictions": np.random.rand(10, 1, 1),
        "targets": np.random.rand(10, 1, 1),
        "unit_id": np.repeat(np.arange(5), 2),
    }
    eval.update(data)
    for unit_id in range(5):
        assert unit_id in eval.unit_metrics


def test_update_dual_metrics_when_scaling_on(multiunit_evaluator_with_scaling_fixture):
    """Tests that metrics are tracked in both original and scaled spaces."""
    eval = multiunit_evaluator_with_scaling_fixture
    preds = np.random.rand(2, 1, 1)
    targets = np.random.rand(2, 1, 1)
    model_out = {"predictions": preds, "targets": targets, "unit_id": np.array([1, 1])}

    eval.update(model_out)

    assert 1 in eval.unit_metrics
    assert 1 in eval.normalized_unit_metrics


def test_compute_aggregates_means_and_logs_per_unit(multiunit_evaluator_fixture):
    """Tests that compute results include per-unit and mean metrics."""
    eval = multiunit_evaluator_fixture
    data = {
        "predictions": np.random.rand(10, 1, 1),
        "targets": np.random.rand(10, 1, 1),
        "unit_id": np.repeat(np.arange(2), 5),
    }
    eval.update(data)
    results = eval.compute(mode="val", epoch=0, step=0)

    assert "mse_0_normalized" in results
    assert "mse_normalized_mean" in results


@patch("lmetk.evaluator.multiunit.plt")
def test_plotting_triggers(mock_plt, multiunit_evaluator_with_plotting_fixture):
    """
    Verifies that plotting logic is invoked when all conditions are met:
    - log_image=True
    - Data exists
    - Logger exists
    """
    eval = multiunit_evaluator_with_plotting_fixture
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # 1. Populate data so guard 'if not self.buffer.preds' passes
    data = {
        "predictions": np.random.rand(10, 1, 1),
        "targets": np.random.rand(10, 1, 1),
        "unit_id": np.ones(10, dtype=int),
    }
    eval.update(data)

    # 2. Mock logger methods to ensure no errors downstream
    eval.log_plot = MagicMock()

    # 3. Call compute (which triggers plotting in the override)
    eval.compute(mode="val", epoch=0, step=0)

    assert mock_plt.subplots.called


def test_prepare_predictions_adds_unit_ids(multiunit_evaluator_fixture):
    """Tests that unit IDs are correctly formatted for NetCDF saving."""
    eval = multiunit_evaluator_fixture
    data = {
        "predictions": np.random.rand(2, 1, 1),
        "targets": np.random.rand(2, 1, 1),
        "unit_id": np.array([1, 2]),
    }
    eval.update(data)
    result = eval._prepare_predictions_for_save()

    assert "unit_ids" in result
    assert isinstance(result["unit_ids"], xr.DataArray)


def test_update_splits_data_by_unit_2d(multiunit_evaluator_fixture):
    """Tests handling of composite (2D) unit IDs (e.g. DatasetID + AssetID)."""
    eval = multiunit_evaluator_fixture
    u_ids = np.array([[1, 101], [1, 102], [2, 101]])
    data = {
        "predictions": np.random.rand(3, 1, 1),
        "targets": np.random.rand(3, 1, 1),
        "unit_id": u_ids,
    }
    eval.update(data)
    assert (1, 101) in eval.unit_metrics
    assert (1, 102) in eval.unit_metrics
    assert (2, 101) in eval.unit_metrics


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_skips_when_no_logger(mock_plt, multiunit_evaluator_factory):
    """Tests that plotting is bypassed if no remote logger is provided."""
    eval = multiunit_evaluator_factory(
        log_image=True, remote_logger=None, use_mock_logger=False
    )
    data = {
        "predictions": np.random.rand(5, 1, 1),
        "targets": np.random.rand(5, 1, 1),
        "unit_id": np.ones(5, dtype=int),
    }
    eval.update(data)
    eval.compute(mode="val", epoch=0, step=0)
    mock_plt.subplots.assert_not_called()


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_handles_empty_predictions(
    mock_plt, multiunit_evaluator_with_plotting_fixture
):
    """Tests that the plotter exits early and safely if no data is present."""
    eval = multiunit_evaluator_with_plotting_fixture
    eval.reset()
    eval._log_pred_vs_target_curves(mode="val", step=0)
    mock_plt.subplots.assert_not_called()
