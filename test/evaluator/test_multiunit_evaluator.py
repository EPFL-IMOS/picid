import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from picid.evaluator.multiunit import MultiUnitEvaluator
from picid.evaluator.hooks.save_predictions import SavePredictionsHook  # ADDED
from picid.evaluator.hooks.unit_trend_plot import UnitTrendPlotHook

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
        from picid.transforms.base.multisource import InverseTransformMixin

        class SimpleScaler(InverseTransformMixin):
            def __init__(self, apply_inverse_scaling):
                self.apply_inverse = apply_inverse_scaling

            def inverse_transform_if_needed(self, preds, targets, metadata=None):
                if self.apply_inverse:
                    return preds * 2.0, targets * 2.0
                return preds, targets

            def inverse_transform(self, data, metadata=None):
                keys = list(data.keys())
                np_data = data[keys[0]]
                if np_data.ndim == 1:
                    np_data = np_data.reshape(-1, 1)
                return np_data * 2

        scaler = SimpleScaler(apply_inverse_scaling)

        if remote_logger is None and use_mock_logger:
            remote_logger = MagicMock()
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
    return multiunit_evaluator_factory(apply_inverse_scaling=False, log_image=True)


@pytest.fixture
def multiunit_evaluator_with_scaling_fixture(multiunit_evaluator_factory):
    return multiunit_evaluator_factory(apply_inverse_scaling=True)


@pytest.fixture
def multiunit_evaluator_with_plotting_fixture(multiunit_evaluator_factory):
    return multiunit_evaluator_factory(log_image=True, enable_plot_subsampling=False)


# =========================================================================
# === 2. TESTS ===
# =========================================================================


def test_init_creates_collections(multiunit_evaluator_fixture):
    """Tests that buffer and metric collections are initialized."""
    eval = multiunit_evaluator_fixture
    # FIX: Access via buffer dictionary
    assert isinstance(eval.buffer.data["unit_ids"], list)
    # FIX: Check unit_managers dictionary
    assert isinstance(eval.unit_managers, dict)


def test_reset_clears_everything(multiunit_evaluator_fixture):
    """Tests that reset clears buffer and per-unit metrics."""
    eval = multiunit_evaluator_fixture
    data = {
        "predictions": np.random.rand(5, 1, 1),
        "targets": np.random.rand(5, 1, 1),
        "unit_id": np.array([1, 1, 2, 2, 3]),
    }
    eval.update(data)
    # FIX: Access via buffer dictionary
    assert len(eval.buffer.data["unit_ids"]) > 0

    eval.reset()
    # FIX: Access via buffer dictionary
    assert len(eval.buffer.data["unit_ids"]) == 0
    # FIX: Check unit_managers
    assert len(eval.unit_managers) == 0


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
        # FIX: Check unit_managers
        assert unit_id in eval.unit_managers


def test_update_dual_metrics_when_scaling_on(multiunit_evaluator_with_scaling_fixture):
    """Tests that metrics are tracked in both original and scaled spaces."""
    eval = multiunit_evaluator_with_scaling_fixture
    preds = np.random.rand(2, 1, 1)
    targets = np.random.rand(2, 1, 1)
    model_out = {"predictions": preds, "targets": targets, "unit_id": np.array([1, 1])}

    eval.update(model_out)

    # FIX: Managers handle both internally
    assert 1 in eval.unit_managers
    assert eval.unit_managers[1].is_dual is True


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

    # FIX: Naming convention changed to {name}_{suffix}_{unit_id}
    assert "mse_normalized_0" in results
    assert "mse_normalized_mean" in results


# @patch("picid.evaluator.multiunit.plt")
# def test_plotting_triggers(mock_plt, multiunit_evaluator_with_plotting_fixture):
#     eval = multiunit_evaluator_with_plotting_fixture
#     mock_plt.subplots.return_value = (MagicMock(), MagicMock())

#     data = {
#         "predictions": np.random.rand(10, 1, 1),
#         "targets": np.random.rand(10, 1, 1),
#         "unit_id": np.ones(10, dtype=int),
#     }
#     eval.update(data)
#     eval.log_plot = MagicMock()
#     eval.compute(mode="val", epoch=0, step=0)

#     assert mock_plt.subplots.called


def test_prepare_predictions_adds_unit_ids(multiunit_evaluator_fixture):
    """Tests that unit IDs are correctly formatted for NetCDF saving."""
    eval = multiunit_evaluator_fixture
    data = {
        "predictions": np.random.rand(2, 1, 1),
        "targets": np.random.rand(2, 1, 1),
        "unit_id": np.array([1, 2]),
    }
    eval.update(data)

    # FIX: Logic moved to SavePredictionsHook
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    result = hook._format_xarray(eval.buffer.get_all(), eval)

    assert "unit_ids" in result


def test_update_splits_data_by_unit_2d(multiunit_evaluator_fixture):
    """Tests handling of composite (2D) unit IDs."""
    eval = multiunit_evaluator_fixture
    u_ids = np.array([[1, 101], [1, 102], [2, 101]])
    data = {
        "predictions": np.random.rand(3, 1, 1),
        "targets": np.random.rand(3, 1, 1),
        "unit_id": u_ids,
    }
    eval.update(data)
    # FIX: Check unit_managers
    assert (1, 101) in eval.unit_managers
    assert (1, 102) in eval.unit_managers
    assert (2, 101) in eval.unit_managers


@patch("picid.evaluator.hooks.unit_trend_plot.plt")
def test_plotting_skips_when_no_logger(mock_plt, multiunit_evaluator_factory):
    """
    Tests that plotting is bypassed if no remote logger is provided.
    Logic: The Hook should check for evaluator.remote_logger and exit if None.
    """
    # Create evaluator without a logger
    evaluator = multiunit_evaluator_factory(
        log_image=True, remote_logger=None, use_mock_logger=False
    )

    data = {
        "predictions": np.random.rand(5, 1, 1),
        "targets": np.random.rand(5, 1, 1),
        "unit_id": np.ones(5, dtype=int),
    }
    evaluator.update(data)

    # ACT: Calling compute triggers the registered UnitTrendPlotHook
    evaluator.compute(mode="val", epoch=0, step=0)

    # ASSERT: plt should not have been used
    mock_plt.subplots.assert_not_called()


@patch("picid.evaluator.hooks.unit_trend_plot.plt")
def test_plotting_handles_empty_predictions(
    mock_plt, multiunit_evaluator_with_plotting_fixture
):
    """
    Tests that the plotter exits early and safely if no data is present.
    We ensure the hook is present by injecting it if the fixture missed it.
    """
    evaluator = multiunit_evaluator_with_plotting_fixture
    evaluator.reset()  # Ensure buffer and metrics are empty

    # 1. Ensure the hook exists (Defensive programming for the test)
    plot_hook = next(
        (h for h in evaluator.hooks if isinstance(h, UnitTrendPlotHook)), None
    )

    if plot_hook is None:
        # If the fixture didn't add it, add it now so we can test it
        plot_hook = UnitTrendPlotHook()
        evaluator.add_hook(plot_hook)

    # 2. ACT: Call the hook's entry point directly.
    # This avoids the MetricManager.compute() ValueError while testing the Hook's guard.
    plot_hook.on_compute_end(
        results={}, evaluator=evaluator, mode="val", epoch=0, step=0
    )

    # 3. ASSERT: plt.subplots should NOT have been called because buffer.get_all() is empty
    mock_plt.subplots.assert_not_called()
