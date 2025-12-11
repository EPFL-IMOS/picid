import pytest
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
        inverse_transform=mock_inverse_transform_mixin,
        apply_inverse_scaling=False,
        save_predictions=False,
        collect_predictions=False,
        paths=MagicMock(),
    )
    evaluator.scaling_wrapper = mock_inverse_transform_mixin
    evaluator.metrics = metrics
    evaluator.metric_names = ["mse", "rmse"]

    # Initialize secondary metrics manually for testing purposes
    evaluator.normalized_metrics = {"mse": MSEMetric(), "rmse": RMSEMetric()}

    return evaluator


@pytest.fixture
def default_evaluator_regression_fixture_saving(default_evaluator_regression_fixture):
    """
    Extends the base fixture to enable saving and collection.
    """
    evaluator = default_evaluator_regression_fixture
    evaluator.save_predictions = True
    evaluator.collect_predictions = True
    evaluator.paths = MagicMock(eval_details="/mock/base")
    evaluator.out_path = None
    # Reset collections
    evaluator.preds, evaluator.targets = [], []
    evaluator.scaled_preds, evaluator.scaled_targets = [], []
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
    mocker.patch.object(
        MetricFactory, "create_metric", side_effect=lambda name, paths: MagicMock()
    )

    # Case A: Scaling OFF
    evaluator_a = DefaultEvaluator(
        metric_names=["m"],
        apply_inverse_scaling=False,
        collect_predictions=True,
        inverse_transform=mock_inverse_transform_mixin,
    )
    assert isinstance(evaluator_a.preds, list)
    assert not hasattr(evaluator_a, "normalized_metrics")

    # Case B: Scaling ON
    evaluator_b = DefaultEvaluator(
        metric_names=["m"],
        apply_inverse_scaling=True,
        collect_predictions=True,
        inverse_transform=mock_inverse_transform_mixin,
    )
    assert hasattr(evaluator_b, "normalized_metrics")
    assert len(evaluator_b.normalized_metrics) == 1


def test_evaluator_reset_clears_all_collections_and_metrics(
    default_evaluator_regression_fixture_saving,
):
    """Tests reset clears all metrics and collected data."""
    evaluator = default_evaluator_regression_fixture_saving
    evaluator.scaling_wrapper.apply_inverse = True
    # Metric instances are already set up in the fixture

    # Setup state
    _update_evaluator(evaluator, P_NORM, T_NORM)
    assert evaluator.metrics["mse"].total_count == 2

    # ACT
    evaluator.reset()

    # ASSERT
    assert len(evaluator.preds) == 0
    assert len(evaluator.scaled_preds) == 0
    assert evaluator.metrics["mse"].total_count == 0
    assert evaluator.normalized_metrics["mse"].total_count == 0


@pytest.mark.parametrize("apply_inverse", [False, True])
def test_update_collection_logic(
    default_evaluator_regression_fixture_saving, apply_inverse
):
    """
    Tests data collection logic when scaling is ON/OFF.
    """
    # Use the fixture instance directly; do not re-instantiate.
    evaluator = default_evaluator_regression_fixture_saving

    # Configure State
    evaluator.scaling_wrapper.apply_inverse = apply_inverse

    # Ensure collections are clean before test
    evaluator.reset()

    P_IN_SAFE = np.array([[[5.0]], [[6.0]]])

    # ACT
    _update_evaluator(evaluator, P_IN_SAFE, P_IN_SAFE)

    # ASSERT
    if apply_inverse:
        # self.preds holds NORMALIZED data
        assert_array_equal(evaluator.preds[0], P_IN_SAFE)
        # self.scaled_preds holds DENORMALIZED data (2x)
        assert_array_equal(evaluator.scaled_preds[0], P_IN_SAFE * 2)
    else:
        # Only self.preds holds data
        assert_array_equal(evaluator.preds[0], P_IN_SAFE)
        assert len(evaluator.scaled_preds) == 0


def test_default_evaluator_dual_reporting_with_scaling(
    default_evaluator_regression_fixture,
):
    """
    Tests Dual Reporting: Scaling is ON. Reports both NORMALIZED and DENORMALIZED.
    """
    evaluator = default_evaluator_regression_fixture
    evaluator.scaling_wrapper.apply_inverse = True

    # ACT
    _update_evaluator(evaluator, P_NORM, T_NORM)

    # COMPUTE
    results = evaluator._compute_metrics()

    # ASSERT
    assert_almost_equal(results["mse_denormalized"], 4.0)  # Scaled 2x -> Error 4.0
    assert_almost_equal(results["mse_normalized"], 1.0)  # Original -> Error 1.0


def test_default_evaluator_compute_no_scaling_correct_naming(
    default_evaluator_regression_fixture,
):
    """
    Tests that when scaling is OFF, the single result is correctly named as _normalized.
    """
    evaluator = default_evaluator_regression_fixture
    evaluator.scaling_wrapper.apply_inverse = False

    # ACT
    _update_evaluator(evaluator, P_NORM, T_NORM)

    # COMPUTE
    results = evaluator._compute_metrics()

    # ASSERT
    assert "mse_denormalized" not in results
    assert "mse_normalized" in results
    assert_almost_equal(results["mse_normalized"], 1.0)


@patch("lmetk.evaluator.default.xr.Dataset")
def test_prepare_predictions_with_scaling(
    mock_xr_dataset, default_evaluator_regression_fixture_saving
):
    """
    Tests _prepare_predictions_for_save correctly handles array concatenation and naming.
    """
    evaluator = default_evaluator_regression_fixture_saving
    evaluator.scaling_wrapper.apply_inverse = True

    # Manually load data
    evaluator.preds.append(np.array([[1], [2]]))
    evaluator.targets.append(np.array([[10], [20]]))
    evaluator.scaled_preds.append(np.array([[100], [200]]))
    evaluator.scaled_targets.append(np.array([[1000], [2000]]))

    # ACT
    dict_to_save = evaluator._prepare_predictions_for_save()

    # ASSERT
    assert_array_equal(dict_to_save["preds"][1], np.array([[100], [200]]))
    assert_array_equal(dict_to_save["preds_normalized"][1], np.array([[1], [2]]))


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
