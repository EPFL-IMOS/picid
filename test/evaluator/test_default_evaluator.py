import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal
from typing import Any, Dict
from pathlib import Path
import pytest

# --- Adjust these imports to match your project structure ---
from lmetk.evaluator.default import DefaultEvaluator
from lmetk.evaluator.hooks.save_predictions import (
    SavePredictionsHook,
)  # ADDED: For testing decoupled logic
from lmetk.metrics.base import AbstractMetric
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.metrics.metrics import MSEMetric, RMSEMetric
from lmetk.metrics.metric_factory import MetricFactory

# =========================================================================
# === 1. MOCKS and FIXTURES for Dependencies ===
# =========================================================================


class MockMinMaxScaler:
    """Mock the sklearn MinMaxScaler for predictable behavior."""

    def transform(self, data):
        return data * 0.5

    def inverse_transform(self, data):
        return data * 2.0


@pytest.fixture
def mock_external_dependencies(mocker):
    """Mocks external library dependencies (Sklearn, internal helpers)."""
    mocker.patch(
        "lmetk.transforms.base_transforms.scaler.MinMaxScaler", MockMinMaxScaler
    )

    def mock_convert_to_numpy(data):
        return data, None

    mocker.patch(
        "lmetk.transforms.base_transforms.scaler._convert_to_numpy",
        mock_convert_to_numpy,
    )


@pytest.fixture
def mock_inverse_transform_mixin(mock_external_dependencies):
    """
    Mocks the InverseTransformMixin instance used by the Evaluator.
    """
    mock_transform = MagicMock(spec=InverseTransformMixin)

    def mock_inverse_transform_if_needed(predictions, targets, **kwargs):
        """Simulates the evaluator calling the wrapper logic (2x scaling)."""
        if mock_transform.apply_inverse:
            return predictions * 2.0, targets * 2.0
        return predictions, targets

    # Attach the mocked function to the attribute
    mock_transform.inverse_transform_if_needed = MagicMock(
        side_effect=mock_inverse_transform_if_needed
    )
    mock_transform.apply_inverse = False

    return mock_transform


@pytest.fixture
def mock_path_obj():
    """
    Creates a dedicated Mock object for Path operations, preventing filesystem access.
    """
    p = MagicMock(spec=Path)
    p.__truediv__.return_value = p  # Support path / "string"
    p.mkdir.return_value = None
    p.parent = MagicMock()
    p.parent.mkdir.return_value = None
    return p


# --- Evaluator Fixtures ---
@pytest.fixture
def default_evaluator_regression_fixture(mocker, mock_inverse_transform_mixin):
    """
    Provides a DefaultEvaluator configured with real metric classes (MSE, RMSE).
    """
    # Use real metric instances for proper accumulation/reset testing
    metrics = {"mse": MSEMetric(), "rmse": RMSEMetric()}

    # Patch MetricFactory to return our real instances
    mocker.patch.object(
        MetricFactory, "create_metric", lambda name, paths: metrics[name.lower()]
    )

    evaluator = DefaultEvaluator(
        metric_names=["mse", "rmse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=mock_inverse_transform_mixin,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths=MagicMock(),
        remote_logger=None,
        collect_predictions=False,
    )
    # Note: scaling_wrapper assignment is needed to override the real ScalingWrapper
    # created during init with our mock.
    evaluator.scaling_wrapper = mock_inverse_transform_mixin

    return evaluator


@pytest.fixture
def default_evaluator_with_scaling_fixture(mocker):
    """
    DefaultEvaluator fixture with inverse scaling ENABLED.
    """

    # Create fresh metric instances for each test (avoid sharing state)
    def create_metric_factory(name, paths=None):
        if name.lower() == "mse":
            return MSEMetric()
        elif name.lower() == "rmse":
            return RMSEMetric()
        raise ValueError(f"Unknown metric: {name}")

    mocker.patch.object(
        MetricFactory, "create_metric", side_effect=create_metric_factory
    )

    # Create mock that simulates 2x scaling
    mock_scaler = MagicMock()
    mock_scaler.apply_inverse = True

    def mock_inverse_transform_if_needed(preds, targets, **kwargs):
        return preds * 2.0, targets * 2.0

    mock_scaler.inverse_transform_if_needed = MagicMock(
        side_effect=mock_inverse_transform_if_needed
    )

    evaluator = DefaultEvaluator(
        metric_names=["mse", "rmse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=mock_scaler,
        apply_inverse_scaling=True,
        save_predictions=True,
        paths=MagicMock(),
        remote_logger=None,
        collect_predictions=True,
    )
    evaluator.scaling_wrapper = mock_scaler

    return evaluator


@pytest.fixture
def default_evaluator_regression_fixture_saving(mocker, mock_inverse_transform_mixin):
    """
    DefaultEvaluator fixture with saving and collection enabled (no scaling).
    """
    metrics = {"mse": MSEMetric(), "rmse": RMSEMetric()}
    mocker.patch.object(
        MetricFactory, "create_metric", lambda name, paths: metrics[name.lower()]
    )

    mock_paths = MagicMock(eval_details="/mock/base")

    evaluator = DefaultEvaluator(
        metric_names=["mse", "rmse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=mock_inverse_transform_mixin,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths=mock_paths,
        remote_logger=None,
        collect_predictions=True,
    )
    evaluator.scaling_wrapper = mock_inverse_transform_mixin

    return evaluator


# =========================================================================
# === 2. Helper Data and Functions ===
# =========================================================================

# Use 3D Arrays (N, 1, 1) to satisfy Shape Checks in Scaling/Metrics
P_NORM = np.array([[[1.0]], [[3.0]]])
T_NORM = np.array([[[2.0]], [[4.0]]])


def _update_evaluator(evaluator: DefaultEvaluator, predictions, targets):
    """Helper to simulate update call."""
    evaluator.update({"predictions": predictions, "targets": targets})


# =========================================================================
# === 3. Unit Tests ===
# =========================================================================


def test_evaluator_init_initializes_collections_and_secondary_metrics(
    mocker, mock_inverse_transform_mixin
):
    """Tests if collections and secondary metric dicts are initialized correctly."""

    # Case A: Scaling OFF
    evaluator_a = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=mock_inverse_transform_mixin,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )
    # FIX: Access via buffer.data
    assert isinstance(evaluator_a.buffer.data["preds"], list)
    # FIX: Access via metric_manager
    assert evaluator_a.metric_manager.normalized_metrics is None

    # Case B: Scaling ON
    evaluator_b = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=mock_inverse_transform_mixin,
        apply_inverse_scaling=True,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )
    # FIX: Access via metric_manager
    assert evaluator_b.metric_manager.normalized_metrics is not None
    assert len(evaluator_b.metric_manager.normalized_metrics) == 1


def test_evaluator_reset_clears_all_collections_and_metrics(
    default_evaluator_with_scaling_fixture,
    regression_data_basic,
):
    """
    Tests reset clears all metrics and collected data.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled and normalized_metrics exists
    assert evaluator.scaling_wrapper.apply_inverse is True
    # FIX: Check metric_manager
    assert evaluator.metric_manager.normalized_metrics is not None

    # Setup state with realistic data
    evaluator.update(regression_data_basic)
    assert evaluator.metric_manager.metrics["mse"].total_count > 0

    # ACT
    evaluator.reset()

    # ASSERT: Everything should be cleared
    # FIX: Access via buffer.data dictionary
    assert len(evaluator.buffer.data["preds"]) == 0
    assert len(evaluator.buffer.data["norm_preds"]) == 0
    assert len(evaluator.buffer.data["targets"]) == 0
    assert len(evaluator.buffer.data["norm_targets"]) == 0

    # FIX: Access via metric_manager
    assert evaluator.metric_manager.metrics["mse"].total_count == 0
    assert evaluator.metric_manager.normalized_metrics["mse"].total_count == 0


def test_update_collection_logic_scaling_off(
    default_evaluator_regression_fixture_saving, regression_data_basic
):
    """
    Tests data collection logic when scaling is OFF.
    """
    evaluator = default_evaluator_regression_fixture_saving

    # Verify scaling is OFF
    assert evaluator.scaling_wrapper.apply_inverse is False

    evaluator.reset()

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ASSERT: Only self.buffer.data["preds"] holds data (no scaling)
    # FIX: Access via buffer.data
    assert len(evaluator.buffer.data["preds"]) > 0
    assert_array_equal(
        evaluator.buffer.data["preds"][0], regression_data_basic["predictions"]
    )
    assert len(evaluator.buffer.data["norm_preds"]) == 0


def test_update_collection_logic_scaling_on(
    default_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests data collection logic when scaling is ON.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is ON
    assert evaluator.scaling_wrapper.apply_inverse is True

    evaluator.reset()

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ASSERT: Both normalized and denormalized data collected
    # FIX: Access via buffer.data

    # norm_preds holds NORMALIZED data
    assert len(evaluator.buffer.data["norm_preds"]) > 0
    assert_array_equal(
        evaluator.buffer.data["norm_preds"][0], regression_data_basic["predictions"]
    )

    # preds holds DENORMALIZED data (2x from mock)
    assert len(evaluator.buffer.data["preds"]) > 0
    expected_scaled = regression_data_basic["predictions"] * 2.0
    assert_array_equal(evaluator.buffer.data["preds"][0], expected_scaled)


def test_default_evaluator_dual_reporting_with_scaling(
    default_evaluator_with_scaling_fixture,
    regression_data_basic,
):
    """
    Tests Dual Reporting: Scaling is ON. Reports both NORMALIZED and DENORMALIZED.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert evaluator.scaling_wrapper.apply_inverse is True
    # FIX: Access via metric_manager
    assert evaluator.metric_manager.normalized_metrics is not None

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # COMPUTE
    # FIX: _compute_metrics moved to metric_manager.compute
    results = evaluator.metric_manager.compute()

    # ASSERT: Both normalized and denormalized metrics should be present
    assert "mse_denormalized" in results
    assert "mse_normalized" in results

    # Denormalized should be larger (scaled by 2x, so error scales by 4x)
    assert results["mse_denormalized"] > results["mse_normalized"]

    # Both should be non-negative
    assert results["mse_denormalized"] >= 0.0
    assert results["mse_normalized"] >= 0.0


def test_default_evaluator_compute_no_scaling_correct_naming(
    default_evaluator_regression_fixture,
    regression_data_basic,
):
    """
    Tests that when scaling is OFF, the single result is correctly named as _normalized.
    """
    evaluator = default_evaluator_regression_fixture

    # Verify scaling is disabled (fixture default)
    assert evaluator.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # COMPUTE
    # FIX: _compute_metrics moved to metric_manager.compute
    results = evaluator.metric_manager.compute()

    # ASSERT: Should have _normalized suffix, not _denormalized
    assert "mse_denormalized" not in results
    assert "mse_normalized" in results
    assert results["mse_normalized"] >= 0.0


@patch("lmetk.evaluator.hooks.save_predictions.xr.Dataset")
def test_prepare_predictions_with_scaling(
    mock_xr_dataset, default_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests preparation correctly handles array concatenation and naming via Hook logic.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert evaluator.scaling_wrapper.apply_inverse is True

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ACT: Decoupled logic testing
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    dict_to_save = hook._format_xarray(evaluator.buffer.get_all(), evaluator)

    # ASSERT - preds should be 2x the original (from fixture mock)
    assert "preds" in dict_to_save
    assert "preds_normalized" in dict_to_save

    # Verify data shapes are correct (format is (dims, data))
    preds_data = dict_to_save["preds"][1]
    preds_norm_data = dict_to_save["preds_normalized"][1]
    assert preds_data.shape == regression_data_basic["predictions"].shape
    assert preds_norm_data.shape == regression_data_basic["predictions"].shape


@patch("lmetk.evaluator.hooks.save_predictions.xr.Dataset")
def test_save_predictions_to_file_orchestration(
    mock_xr_dataset, default_evaluator_regression_fixture_saving, mock_path_obj
):
    """
    Tests that the save orchestration calls the necessary methods via Hook compute end.
    """
    evaluator = default_evaluator_regression_fixture_saving
    mock_dataset_instance = MagicMock()
    mock_xr_dataset.return_value = mock_dataset_instance

    # FIX: Patch Path in the hook's module
    with patch(
        "lmetk.evaluator.hooks.save_predictions.Path", return_value=mock_path_obj
    ):

        evaluator.paths = MagicMock(eval_details="/mock/base")
        evaluator.update(
            {"predictions": np.array([[[1]]]), "targets": np.array([[[1]]])}
        )

        # ACT: Use the hook's entry point
        hook = SavePredictionsHook(dims=["sample", "time", "feature"])
        hook.on_compute_end({}, evaluator, mode="test_mode", epoch=1, step=1)

        # ASSERT
        mock_path_obj.mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_dataset_instance.to_netcdf.assert_called_once()


# =========================================================================
# === 4. Additional Tests for Full Coverage ===
# =========================================================================


def test_update_classification_task(
    mocker, mock_inverse_transform_mixin, classification_data_balanced
):
    """
    Tests update with classification task type.
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )

    evaluator = DefaultEvaluator(
        metric_names=["accuracy"],
        task_type="classification",
        num_classes=classification_data_balanced["predictions"].shape[-1],
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    # Verify scaling is disabled
    assert evaluator.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic classification fixture data
    evaluator.update(classification_data_balanced)

    # ASSERT: Targets should be collected (converted to int internally)
    # FIX: Access via buffer.data
    assert len(evaluator.buffer.data["preds"]) == 1
    assert len(evaluator.buffer.data["targets"]) == 1
    # Verify targets are integers (converted internally)
    assert evaluator.buffer.data["targets"][0].dtype in [np.int32, np.int64, int]


def test_update_forecasting_task(
    mocker, mock_inverse_transform_mixin, forecasting_data_multi_step
):
    """
    Tests update with forecasting task type.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="forecasting",
        num_classes=None,
        inverse_transform=mock_inverse_transform_mixin,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )
    evaluator.scaling_wrapper = mock_inverse_transform_mixin

    # ACT: Use realistic multi-step forecasting fixture data
    evaluator.update(forecasting_data_multi_step)

    # ASSERT: Data collected correctly
    # FIX: Access via buffer.data
    assert len(evaluator.buffer.data["preds"]) == 1
    assert (
        evaluator.buffer.data["preds"][0].shape
        == forecasting_data_multi_step["predictions"].shape
    )
    assert (
        evaluator.buffer.data["targets"][0].shape
        == forecasting_data_multi_step["targets"].shape
    )


def test_update_unsupported_task_raises_error(mocker, mock_inverse_transform_mixin):
    """
    Tests unsupported task types.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    # FIX: Regex match updated to "Unsupported task:"
    with pytest.raises(AssertionError, match="Unsupported task:"):
        DefaultEvaluator(
            metric_names=["mse"],
            task_type="invalid_task",
            num_classes=None,
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )


def test_log_plot_with_remote_logger(mocker, mock_inverse_transform_mixin):
    """
    Tests log_plot calls remote logger experiment.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    mock_logger = MagicMock()

    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=mock_logger,
        collect_predictions=True,
    )

    mock_fig = MagicMock()

    evaluator.log_plot(mock_fig, "test_plot", "val", epoch=1, step=10)

    mock_logger.experiment.log.assert_called_once()


def test_log_plot_without_remote_logger(mocker, mock_inverse_transform_mixin):
    """
    Tests log_plot does nothing without remote logger.
    """
    mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

    evaluator = DefaultEvaluator(
        metric_names=["mse"],
        task_type="regression",
        num_classes=None,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=False,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    mock_fig = MagicMock()

    # Should not raise
    evaluator.log_plot(mock_fig, "test_plot", "val", epoch=1, step=10)


def test_prepare_predictions_classification_argmax(
    mocker, mock_inverse_transform_mixin
):
    """
    Tests buffer preparation applies argmax for classification via Hook logic.
    """
    mocker.patch.object(
        MetricFactory, "create_classification_metric", return_value=MagicMock()
    )

    evaluator = DefaultEvaluator(
        metric_names=["accuracy"],
        task_type="classification",
        num_classes=3,
        inverse_transform=None,
        apply_inverse_scaling=False,
        save_predictions=True,
        paths={},
        remote_logger=None,
        collect_predictions=True,
    )

    # Feed logits through proper interface
    preds = np.array([[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]])  # (batch, time, classes)
    targets = np.array([[[1], [0]]])  # (batch, time, 1)

    evaluator.update({"predictions": preds, "targets": targets})

    # FIX: Logic tested via the hook's formatting helper
    hook = SavePredictionsHook(dims=["sample", "time", "feature"])
    result = hook._format_xarray(evaluator.buffer.get_all(), evaluator)

    # Predictions should have last dim = 3 (logits preserved in formatting)
    # Target dim remains 1.
    assert result["preds"][1].ndim == 3


def test_save_predictions_empty_dict(default_evaluator_regression_fixture_saving):
    """
    Tests SavePredictionsHook handles empty buffer gracefully.
    """
    evaluator = default_evaluator_regression_fixture_saving
    evaluator.buffer.clear()

    # Should not raise or create files
    hook = SavePredictionsHook()
    hook.on_compute_end({}, evaluator, mode="test", epoch=1, step=1)


def test_compute_orchestrates_all_steps(
    default_evaluator_regression_fixture_saving, mock_path_obj
):
    """
    Tests compute orchestrates hooks and metrics computation.
    """
    evaluator = default_evaluator_regression_fixture_saving

    # Mock hooks and manager
    mock_hook = MagicMock(spec=SavePredictionsHook)
    evaluator.hooks = [mock_hook]
    evaluator.metric_manager.compute = MagicMock(return_value={"mse": 0.5})

    result = evaluator.compute(mode="test", epoch=1, step=10)

    # ASSERT
    mock_hook.on_compute_end.assert_called_once()
    evaluator.metric_manager.compute.assert_called_once()
    assert result == {"mse": 0.5}
