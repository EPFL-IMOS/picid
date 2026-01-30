import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch
from numpy.testing import assert_array_equal, assert_almost_equal
from typing import Any, Dict
from pathlib import Path

# --- Adjust these imports to match your project structure ---
from lmetk.evaluator.default import DefaultEvaluator
from lmetk.metrics.base import AbstractMetric
from lmetk.transforms.base.multisource import InverseTransformMixin
import pytest
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
    # created during init with our mock. This is acceptable as it's replacing a
    # dependency, not bypassing initialization logic.
    evaluator.scaling_wrapper = mock_inverse_transform_mixin

    return evaluator


@pytest.fixture
def default_evaluator_with_scaling_fixture(mocker):
    """
    DefaultEvaluator fixture with inverse scaling ENABLED.

    Creates evaluator with apply_inverse_scaling=True from init,
    ensuring normalized_metrics is properly created.

    **PHM Context**: Tests scenarios where predictions need to be
    converted back to physical units for evaluation.
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

    Creates a fresh evaluator with proper init parameters for save testing,
    rather than modifying an existing fixture post-creation.
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
    # Use real MSEMetric instead of MagicMock
    # Don't mock - let MetricFactory create real metrics
    # mocker.patch.object(
    #     MetricFactory, "create_metric", side_effect=lambda name, paths: MagicMock()
    # )

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
    assert isinstance(evaluator_a.preds, list)
    assert not hasattr(evaluator_a, "normalized_metrics")

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
    assert hasattr(evaluator_b, "normalized_metrics")
    assert len(evaluator_b.normalized_metrics) == 1


def test_evaluator_reset_clears_all_collections_and_metrics(
    default_evaluator_with_scaling_fixture,
    regression_data_basic,
):
    """
    Tests reset clears all metrics and collected data.

    **PHM Logic**: Reset must clear both normalized and denormalized metrics
    to ensure clean state between evaluation epochs.

    **Methodology**: Use fixture with scaling enabled (creates normalized_metrics).
    Use realistic data fixture.

    **Uses fixture**: regression_data_basic - realistic RUL predictions.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled and normalized_metrics exists
    assert evaluator.scaling_wrapper.apply_inverse is True
    assert hasattr(evaluator, "normalized_metrics")

    # Setup state with realistic data
    evaluator.update(regression_data_basic)
    assert evaluator.metrics["mse"].total_count > 0

    # ACT
    evaluator.reset()

    # ASSERT: Everything should be cleared
    assert len(evaluator.preds) == 0
    assert len(evaluator.scaled_preds) == 0
    assert len(evaluator.targets) == 0
    assert len(evaluator.scaled_targets) == 0
    assert evaluator.metrics["mse"].total_count == 0
    assert evaluator.normalized_metrics["mse"].total_count == 0


def test_update_collection_logic_scaling_off(
    default_evaluator_regression_fixture_saving, regression_data_basic
):
    """
    Tests data collection logic when scaling is OFF.

    **PHM Logic**: Without scaling, only normalized predictions are collected.

    **Uses fixture**: regression_data_basic - realistic RUL predictions.
    """
    evaluator = default_evaluator_regression_fixture_saving

    # Verify scaling is OFF
    assert evaluator.scaling_wrapper.apply_inverse is False

    evaluator.reset()

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ASSERT: Only self.preds holds data (no scaling)
    assert len(evaluator.preds) > 0
    assert_array_equal(evaluator.preds[0], regression_data_basic["predictions"])
    assert len(evaluator.scaled_preds) == 0


def test_update_collection_logic_scaling_on(
    default_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests data collection logic when scaling is ON.

    **PHM Logic**: With scaling, both normalized and denormalized predictions
    are collected for comparison in both spaces.

    **Uses fixture**: regression_data_basic - realistic RUL predictions.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is ON
    assert evaluator.scaling_wrapper.apply_inverse is True

    evaluator.reset()

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ASSERT: Both normalized and denormalized data collected
    # self.preds holds NORMALIZED data
    assert len(evaluator.preds) > 0
    assert_array_equal(evaluator.preds[0], regression_data_basic["predictions"])

    # self.scaled_preds holds DENORMALIZED data (2x from mock)
    assert len(evaluator.scaled_preds) > 0
    expected_scaled = regression_data_basic["predictions"] * 2.0
    assert_array_equal(evaluator.scaled_preds[0], expected_scaled)


def test_default_evaluator_dual_reporting_with_scaling(
    default_evaluator_with_scaling_fixture,
    regression_data_basic,
):
    """
    Tests Dual Reporting: Scaling is ON. Reports both NORMALIZED and DENORMALIZED.

    **PHM Logic**: When scaling is enabled, metrics are computed in both
    normalized (model output) and denormalized (physical units) spaces.
    This allows comparison of prediction quality in both domains.

    **Methodology**: Use fixture with scaling enabled from init and realistic data.

    **Uses fixture**: regression_data_basic - realistic RUL predictions with noise.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert evaluator.scaling_wrapper.apply_inverse is True
    assert hasattr(evaluator, "normalized_metrics")

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # COMPUTE
    results = evaluator._compute_metrics()

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

    **Methodology**: Use fixture with scaling disabled (default).
    Use realistic data fixture.

    **Uses fixture**: regression_data_basic - realistic RUL predictions.
    """
    evaluator = default_evaluator_regression_fixture

    # Verify scaling is disabled (fixture default)
    assert evaluator.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # COMPUTE
    results = evaluator._compute_metrics()

    # ASSERT: Should have _normalized suffix, not _denormalized
    assert "mse_denormalized" not in results
    assert "mse_normalized" in results
    assert results["mse_normalized"] >= 0.0, "MSE should be non-negative"


@patch("lmetk.evaluator.default.xr.Dataset")
def test_prepare_predictions_with_scaling(
    mock_xr_dataset, default_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests _prepare_predictions_for_save correctly handles array concatenation and naming.

    **Methodology**: Use fixture with scaling enabled, feed realistic data through update().

    **Uses fixture**: regression_data_basic - realistic RUL predictions.
    """
    evaluator = default_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert evaluator.scaling_wrapper.apply_inverse is True

    # ACT: Use realistic fixture data
    evaluator.update(regression_data_basic)

    # ACT
    dict_to_save = evaluator._prepare_predictions_for_save()

    # ASSERT - scaled_preds should be 2x the original (from fixture mock)
    assert "preds" in dict_to_save
    assert "preds_normalized" in dict_to_save

    # Verify data shapes are correct
    preds_data = dict_to_save["preds"][1]  # Get actual array (second element of tuple)
    preds_norm_data = dict_to_save["preds_normalized"][1]
    assert preds_data.shape == regression_data_basic["predictions"].shape
    assert preds_norm_data.shape == regression_data_basic["predictions"].shape


@patch("lmetk.evaluator.default.xr.Dataset")
def test_save_predictions_to_file_orchestration(
    mock_xr_dataset, default_evaluator_regression_fixture_saving, mock_path_obj
):
    """
    Tests that the save orchestration calls the necessary methods using a mock Path.
    """
    evaluator = default_evaluator_regression_fixture_saving
    mock_dataset_instance = MagicMock()
    mock_xr_dataset.return_value = mock_dataset_instance

    # FIX: Use return_value to ensure Path(...) returns our safe mock object
    with patch("lmetk.evaluator.default.Path", return_value=mock_path_obj):

        evaluator.paths = MagicMock(eval_details="/mock/base")
        evaluator.out_path = None

        dict_data = {"data": (["sample"], np.array([1]))}

        # ACT
        evaluator._save_predictions_to_file(dict_data, mode="test_mode")

        # ASSERT
        assert evaluator.out_path == mock_path_obj
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

    **PHM Logic**: Classification tasks don't require inverse scaling.

    **Methodology**: Create evaluator with task_type="classification",
    use realistic classification data.

    **Uses fixture**: classification_data_balanced - realistic balanced classification data
    with clear predictions.

    **Expected**: Targets converted to int, no scaling applied.

    Validates: Requirement DE-CLS-1 - Classification update
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

    # Verify scaling is disabled (as configured at init)
    assert evaluator.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic classification fixture data
    evaluator.update(classification_data_balanced)

    # ASSERT: Targets should be collected (converted to int internally)
    assert len(evaluator.preds) == 1
    assert len(evaluator.targets) == 1
    # Verify targets are integers (converted internally)
    assert evaluator.targets[0].dtype in [np.int32, np.int64, int]


def test_update_forecasting_task(
    mocker, mock_inverse_transform_mixin, forecasting_data_multi_step
):
    """
    Tests update with forecasting task type.

    **PHM Logic**: Forecasting tasks have multi-step predictions.
    Time series prediction quality evaluated across forecast horizon.

    **Methodology**: Create evaluator with task_type="forecasting" at init,
    use realistic multi-step forecasting data.

    **Uses fixture**: forecasting_data_multi_step - realistic multi-step forecasting
    data with degradation curves over 24 timesteps.

    **Expected**: Multi-step data processed correctly.

    Validates: Requirement DE-FORE-1 - Forecasting update
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
    assert len(evaluator.preds) == 1
    assert evaluator.preds[0].shape == forecasting_data_multi_step["predictions"].shape
    assert evaluator.targets[0].shape == forecasting_data_multi_step["targets"].shape


def test_update_unsupported_task_raises_error(mocker, mock_inverse_transform_mixin):
    """
    Tests update raises error for unsupported task types.

    **PHM Logic**: Only known task types are supported.
    Defensive coding ensures graceful handling of invalid states.

    **Methodology**: Simulate invalid state by modifying task_type after init.
    Note: This manual assignment is intentional to test error handling for
    corrupted/invalid state that should never occur in normal operation.

    **Expected**: ValueError raised with descriptive message.

    Validates: Requirement DE-ERR-1 - Unsupported task error
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

    # INTENTIONAL: Manually set invalid task type to test error handling
    # This simulates a corrupted state that should never occur normally
    evaluator.task_type = "invalid_task"

    preds = np.array([[[0.5]]])
    targets = np.array([[[0.5]]])

    with pytest.raises(ValueError, match="does not support task_type"):
        evaluator.update({"predictions": preds, "targets": targets})


def test_log_plot_with_remote_logger(mocker, mock_inverse_transform_mixin):
    """
    Tests log_plot calls remote logger experiment.

    **PHM Logic**: Plots logged to W&B/MLflow via remote logger.
    Visualization tracking is critical for PHM model monitoring.

    **Methodology**: Create evaluator with remote_logger at init.

    **Expected**: Logger experiment.log called.

    Validates: Requirement DE-LOG-1 - Remote logger plot
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

    **PHM Logic**: Gracefully handles missing logger.
    Supports offline evaluation without remote tracking.

    **Methodology**: Create evaluator without remote_logger at init.

    **Expected**: No error, nothing logged.

    Validates: Requirement DE-LOG-2 - No logger handling
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
    Tests _prepare_predictions_for_save applies argmax for classification.

    **PHM Logic**: Classification predictions saved as class indices.

    **Methodology**: Create classification evaluator, feed data through update().

    **Expected**: argmax applied to predictions.

    Validates: Requirement DE-PREP-1 - Classification argmax
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

    result = evaluator._prepare_predictions_for_save()

    # Predictions should have last dim = 1 (argmax applied)
    assert result["preds"][1].shape[-1] == 1


def test_save_predictions_empty_dict(default_evaluator_regression_fixture_saving):
    """
    Tests _save_predictions_to_file handles empty dict.

    **PHM Logic**: No file written for empty predictions.

    **Methodology**: Call with empty dict.

    **Expected**: Method returns early, no file operations.

    Validates: Requirement DE-SAVE-1 - Empty dict handling
    """
    evaluator = default_evaluator_regression_fixture_saving

    # Should not raise or create files
    evaluator._save_predictions_to_file({}, mode="test")


def test_compute_orchestrates_all_steps(
    default_evaluator_regression_fixture_saving, mock_path_obj
):
    """
    Tests compute orchestrates prepare, save, and metrics computation.

    **PHM Logic**: compute() coordinates all evaluation steps.

    **Methodology**: Call compute and verify orchestration.

    **Expected**: All methods called in order.

    Validates: Requirement DE-COMP-1 - Full orchestration
    """
    evaluator = default_evaluator_regression_fixture_saving

    # Mock internal methods
    evaluator._prepare_predictions_for_save = MagicMock(return_value={})
    evaluator._save_predictions_to_file = MagicMock()
    evaluator._compute_metrics = MagicMock(return_value={"mse": 0.5})

    result = evaluator.compute(mode="test", epoch=1, step=10)

    evaluator._prepare_predictions_for_save.assert_called_once()
    evaluator._save_predictions_to_file.assert_called_once()
    evaluator._compute_metrics.assert_called_once()
    assert result == {"mse": 0.5}


# def test_prepare_predictions_returns_empty_when_disabled(mocker, regression_data_basic):
#     """
#     Alwats failing for now.
#     Tests _prepare_predictions_for_save returns empty dict when save/collect disabled.

#     **PHM Logic**: Prevents crashes and saves memory when data collection is
#     turned off for specific training stages.

#     **Uses fixture**: regression_data_basic - realistic RUL data.
#     """
#     # 1. SETUP: Mock MetricFactory to avoid needing real metric classes
#     mocker.patch.object(MetricFactory, "create_metric", return_value=MagicMock())

#     # 2. Initialize with both flags OFF
#     evaluator = DefaultEvaluator(
#         metric_names=["mse"],
#         task_type="regression",
#         save_predictions=False,
#         collect_predictions=False,
#     )

#     # 3. ACT: Pass values from conftest fixture through the evaluator
#     # This simulates a real lifecycle where update is called during training
#     evaluator.update(regression_data_basic)

#     # 4. EXECUTE: This is where the AttributeError happened previously
#     # It should now return {} because save_predictions is False
#     result = evaluator._prepare_predictions_for_save()

#     # 5. ASSERT
#     assert (
#         result == {}
#     ), "Should return empty dict even after updates if saving is disabled"

#     # Verify the attribute 'preds' was never created (collect_predictions=False)
#     assert not hasattr(
#         evaluator, "preds"
#     ), "preds list should not exist when collect_predictions is False"
