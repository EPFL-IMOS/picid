import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, patch, call
from collections import defaultdict
from pathlib import Path

# Adjust imports to match your structure
from lmetk.evaluator.multiunit import MultiUnitEvaluator
from lmetk.metrics.base import AbstractMetric
from lmetk.metrics.metric_factory import MetricFactory

# =========================================================================
# === 1. MOCKS ===
# =========================================================================


class MockMetric(AbstractMetric):
    """Simple mock metric that sums inputs."""

    def __init__(self, name="mock"):
        super().__init__(name)
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, preds, targets):
        self.sum += np.sum(preds)
        self.count += preds.size

    def compute(self):
        return self.sum


@pytest.fixture
def mock_path_obj():
    """Robust path mock preventing filesystem access."""
    p = MagicMock(spec=Path)
    p.__truediv__.return_value = p
    p.mkdir.return_value = None
    return p


@pytest.fixture
def multiunit_evaluator_fixture(mocker, mock_path_obj):
    """
    Sets up a MultiUnitEvaluator with mocked dependencies.
    """
    # Mock Factory to return our simple MockMetric
    mocker.patch.object(
        MetricFactory, "create_metric", side_effect=lambda name, paths: MockMetric(name)
    )

    # Mock Scaler Wrapper (Generic MagicMock to avoid spec issues)
    mock_scaler = MagicMock()

    # Define inverse transform logic: 2x scaling
    def inverse_logic(preds, targets, metadata=None):
        if mock_scaler.apply_inverse:
            return preds * 2.0, targets * 2.0
        return preds, targets

    # Set the side effect on the mock method
    mock_scaler.inverse_transform_if_needed.side_effect = inverse_logic
    mock_scaler.apply_inverse = False  # Default OFF

    # Mock Logger
    mock_logger = MagicMock()

    evaluator = MultiUnitEvaluator(
        metric_names=["sum_metric"],
        inverse_transform=mock_scaler,
        apply_inverse_scaling=False,
        save_predictions=True,
        collect_predictions=True,
        paths=MagicMock(eval_details=mock_path_obj),
        remote_logger=mock_logger,
        log_per_unit_metrics=True,
    )
    # Manually attach the wrapper mock
    evaluator.scaling_wrapper = mock_scaler

    return evaluator


# =========================================================================
# === 2. Tests: Initialization & Reset ===
# =========================================================================


def test_init_creates_collections(multiunit_evaluator_fixture):
    """Tests that unit_ids collection and metric dicts are initialized."""
    eval = multiunit_evaluator_fixture
    assert isinstance(eval.unit_ids, list)
    assert isinstance(eval.unit_metrics, defaultdict)
    # Should be empty initially
    assert len(eval.unit_metrics) == 0


def test_reset_clears_everything(multiunit_evaluator_fixture):
    """Tests that reset wipes data and clears nested metric dictionaries."""
    eval = multiunit_evaluator_fixture

    # Simulate some state
    eval.unit_ids.append(np.array([1]))
    # Manually add a metric for unit 1
    eval.unit_metrics[1]["sum_metric"].update(np.array([10]), np.array([10]))

    eval.reset()

    assert len(eval.unit_ids) == 0
    assert len(eval.unit_metrics) == 0


# =========================================================================
# === 3. Tests: Update Logic (Splitting & Scaling) ===
# =========================================================================


def test_update_splits_data_by_unit_1d(multiunit_evaluator_fixture):
    """
    Tests correct routing of data to specific unit metrics when unit_ids are 1D.
    """
    eval = multiunit_evaluator_fixture

    # Batch of 3 samples: Unit 1, Unit 2, Unit 1
    preds = np.array([[10.0], [20.0], [30.0]])  # Shape (3, 1)
    targets = np.array([[10.0], [20.0], [30.0]])
    unit_ids = np.array([1, 2, 1])  # Shape (3,)

    model_out = {"predictions": preds, "targets": targets, "unit_id": unit_ids}

    # ACT
    eval.update(model_out)

    # ASSERT
    # Unit 1 should have received preds 10.0 and 30.0 -> Sum = 40.0
    assert eval.unit_metrics[1]["sum_metric"].compute() == 40.0

    # Unit 2 should have received preds 20.0 -> Sum = 20.0
    assert eval.unit_metrics[2]["sum_metric"].compute() == 20.0


def test_update_dual_metrics_when_scaling_on(multiunit_evaluator_fixture):
    """
    Tests that when scaling is ON:
    1. Primary metrics get Denormalized data (2x).
    2. Secondary metrics get Normalized data (1x).
    """
    eval = multiunit_evaluator_fixture
    eval.scaling_wrapper.apply_inverse = True

    # FIX 1: Manually create the normalized storage because init skipped it
    eval.normalized_unit_metrics = defaultdict(eval._create_metric_dict)

    # One sample for Unit 5. Value = 10.0.
    preds = np.array([[10.0]])
    targets = preds
    unit_ids = np.array([5])

    model_out = {"predictions": preds, "targets": targets, "unit_id": unit_ids}

    # ACT
    eval.update(model_out)

    # ASSERT
    # Primary (Denormalized): Input 10 * 2 (scaling) = 20
    assert eval.unit_metrics[5]["sum_metric"].compute() == 20.0

    # Secondary (Normalized): Input 10 (original)
    assert eval.normalized_unit_metrics[5]["sum_metric"].compute() == 10.0


# =========================================================================
# === 4. Tests: Compute & Aggregation ===
# =========================================================================


def test_compute_aggregates_means_and_logs_per_unit(multiunit_evaluator_fixture):
    """
    Tests the complex aggregation logic in _compute_metrics.
    Scenario: Scaling OFF.
    """
    eval = multiunit_evaluator_fixture

    # Setup state: Unit 1 value=10, Unit 2 value=30
    eval.unit_metrics[1]["sum_metric"].update(np.array([10]), np.array([0]))
    eval.unit_metrics[2]["sum_metric"].update(np.array([30]), np.array([0]))

    # ACT
    results = eval._compute_metrics()

    # ASSERT
    # 1. Check per-unit logs (keys have _denormalized suffix by default in _compute)
    assert results["sum_metric_1_denormalized"] == 10.0
    assert results["sum_metric_2_denormalized"] == 30.0

    # 2. Check global mean: (10 + 30) / 2 = 20
    assert results["sum_metric_denormalized_mean"] == 20.0


def test_compute_dual_aggregation(multiunit_evaluator_fixture):
    """
    Tests aggregation when both normalized and denormalized metrics exist.
    """
    eval = multiunit_evaluator_fixture
    eval.scaling_wrapper.apply_inverse = True

    # FIX 1: Manually create the normalized storage because init skipped it
    eval.normalized_unit_metrics = defaultdict(eval._create_metric_dict)

    # Setup: Unit 1
    # Denorm = 100, Norm = 10
    eval.unit_metrics[1]["sum_metric"].update(np.array([100]), np.array([0]))
    eval.normalized_unit_metrics[1]["sum_metric"].update(np.array([10]), np.array([0]))

    # Setup: Unit 2
    # Denorm = 200, Norm = 20
    eval.unit_metrics[2]["sum_metric"].update(np.array([200]), np.array([0]))
    eval.normalized_unit_metrics[2]["sum_metric"].update(np.array([20]), np.array([0]))

    # ACT
    results = eval._compute_metrics()

    # ASSERT
    # Mean Denormalized: (100 + 200) / 2 = 150
    assert results["sum_metric_denormalized_mean"] == 150.0

    # Mean Normalized: (10 + 20) / 2 = 15
    assert results["sum_metric_normalized_mean"] == 15.0

    # Check per-unit existence
    assert "sum_metric_1_denormalized" in results
    assert "sum_metric_1_normalized" in results


# =========================================================================
# === 5. Tests: Plotting Side-Effects ===
# =========================================================================


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_triggers(mock_plt, multiunit_evaluator_fixture):
    """
    Tests that plotting logic is triggered for validation steps.
    """
    eval = multiunit_evaluator_fixture
    eval.log_fit_vs_target_curves = True

    # FIX 2: Configure subplots to return a tuple (Figure, Axis)
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # Populate data (needs to be collected first)
    preds = np.zeros((10, 10))
    targets = np.zeros((10, 10))
    unit_ids = np.ones((10,))  # All unit 1

    eval.preds.append(preds)
    eval.targets.append(targets)
    eval.unit_ids.append(unit_ids)

    # FIX 3: Mock the correct method name (log_plot, not _log_plot)
    eval.log_plot = MagicMock()

    # ACT
    # Call compute (which calls super().compute, but triggers plotting first)
    with patch("lmetk.evaluator.default.DefaultEvaluator.compute", return_value={}):
        eval.compute(mode="val", epoch=0, step=0)

    # ASSERT
    # Should have created a figure
    mock_plt.subplots.assert_called()
    # Should have logged it (using the correct mock)
    eval.log_plot.assert_called()
    # Should have closed it
    mock_plt.close.assert_called()


# =========================================================================
# === 6. Tests: Saving (Unit IDs) ===
# =========================================================================


@patch("lmetk.evaluator.default.DefaultEvaluator._prepare_predictions_for_save")
def test_prepare_predictions_adds_unit_ids(
    mock_super_prep, multiunit_evaluator_fixture
):
    """
    Tests that unit_ids are added to the saved dictionary.
    """
    eval = multiunit_evaluator_fixture

    # Mock parent return
    mock_super_prep.return_value = {"preds": "dummy"}

    # Add dummy unit ids
    eval.unit_ids.append(np.array([1, 2]))

    # ACT
    result = eval._prepare_predictions_for_save()

    # ASSERT
    assert "unit_ids" in result
    assert isinstance(result["unit_ids"], xr.DataArray)
    np.testing.assert_array_equal(result["unit_ids"].values, np.array([1, 2]))
