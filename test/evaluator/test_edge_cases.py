"""
Comprehensive edge case tests using fixture data.

This module tests edge cases and boundary conditions using the edge case
fixtures from conftest.py. These tests ensure robust error handling and
graceful degradation.
"""

import pytest
import numpy as np
from picid.evaluator.classification import ClassificationEvaluator
from picid.evaluator.default import DefaultEvaluator
from picid.evaluator.multiunit import MultiUnitEvaluator
from picid.metrics.metric_factory import MetricFactory
from picid.metrics.metrics import MSEMetric

# =========================================================================
# === 1. FIXTURES ===
# =========================================================================


@pytest.fixture
def data_empty_batch():
    """Returns a batch with valid shapes but 0 samples."""
    return {
        "predictions": np.empty((0, 10, 1)),
        "targets": np.empty((0, 10, 1)),
        "unit_id": np.empty((0,), dtype=int),
    }


# =========================================================================
# === 2. TESTS ===
# =========================================================================


class TestEmptyDataHandling:
    def test_empty_batch_default_evaluator(self, data_empty_batch, mocker):
        """
        Test DefaultEvaluator raises ValueError when computing on empty data.
        """
        # Setup: Real MSE metric to trigger the actual ValueError logic
        mocker.patch.object(MetricFactory, "create_metric", return_value=MSEMetric())

        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # 1. Update with empty data (should not crash here, just accumulates nothing)
        evaluator.update(data_empty_batch)

        # 2. Compute should STRICTLY fail because no data was provided
        with pytest.raises(ValueError, match="No data available for computation"):
            evaluator.compute(mode="test", epoch=1, step=1)

    def test_empty_batch_multiunit(self, data_empty_batch):
        """
        Test MultiUnitEvaluator raises ValueError when computing on empty data.

        This aligns behavior with the regression/default evaluator:
        Empty input -> No metrics -> Error on compute.
        """
        # Use real metrics - no mocking to ensure we hit the real exception
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Ensure unit_id matches the empty batch size
        empty_multiunit_data = data_empty_batch.copy()
        empty_multiunit_data["unit_id"] = np.array([], dtype=np.int32)

        # 1. Update with empty data
        evaluator.update(empty_multiunit_data)

        # 2. Assert Strict Failure
        with pytest.raises(ValueError, match="No data available for computation"):
            # MultiUnit delegates to DefaultEvaluator for global metrics,
            # which triggers the ValueError
            evaluator.compute(mode="test", epoch=1, step=1)


class TestPerfectPredictions:
    """Tests with perfect predictions (zero error case)."""

    def test_perfect_predictions_regression(self, data_perfect_predictions):
        """Test that perfect predictions yield zero error metrics."""
        # Use real metrics - no mocking
        evaluator = DefaultEvaluator(
            metric_names=["mse", "mae", "rmse"],
            task_type="regression",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        evaluator.update(data_perfect_predictions)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # All error metrics should be zero (or very close) with real metrics
        assert (
            results["mse_normalized"] < 1e-6
        ), f"MSE should be near zero, got {results['mse_normalized']}"
        assert (
            results["mae_normalized"] < 1e-6
        ), f"MAE should be near zero, got {results['mae_normalized']}"
        assert (
            results["rmse_normalized"] < 1e-6
        ), f"RMSE should be near zero, got {results['rmse_normalized']}"

    def test_perfect_predictions_multiunit(self, data_perfect_predictions):
        """Test multi-unit evaluator with perfect predictions."""
        # Use real metrics - no mocking
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Add unit IDs
        perfect_multiunit_data = data_perfect_predictions.copy()
        perfect_multiunit_data["unit_id"] = np.ones(
            len(perfect_multiunit_data["predictions"]), dtype=np.int32
        )

        evaluator.update(perfect_multiunit_data)
        # FIX: Use public compute() instead of deleted _compute_metrics()
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # Mean should be zero with real metrics
        assert (
            results["mse_normalized_mean"] < 1e-6
        ), f"MSE mean should be near zero, got {results['mse_normalized_mean']}"


class TestExtremeValues:
    """Tests with extreme values (boundary conditions)."""

    def test_extreme_values_regression(self, data_extreme_values):
        """Test evaluator handles extreme values (0.0, 1.0) correctly."""
        # Use real metrics - no mocking
        evaluator = DefaultEvaluator(
            metric_names=["mse", "mae"],
            task_type="regression",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Should not raise
        evaluator.update(data_extreme_values)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # Real metrics should be computed correctly
        assert "mse_normalized" in results
        assert "mae_normalized" in results
        assert (
            results["mse_normalized"] >= 0.0
        ), f"MSE should be non-negative, got {results['mse_normalized']}"
        assert (
            results["mae_normalized"] >= 0.0
        ), f"MAE should be non-negative, got {results['mae_normalized']}"

        # Verify metrics are finite (not NaN or Inf)
        assert np.isfinite(
            results["mse_normalized"]
        ), f"MSE should be finite, got {results['mse_normalized']}"
        assert np.isfinite(
            results["mae_normalized"]
        ), f"MAE should be finite, got {results['mae_normalized']}"

    def test_extreme_values_with_scaling(self, data_extreme_values):
        """Test extreme values with scaling enabled."""
        # Use ConstantScaler(factor=0.5) so inverse_transform divides by 0.5 (= scale by 2)
        from picid.transforms.base_transforms.scaler import ConstantScaler

        scaler = ConstantScaler(factor=0.5)

        # Use real metrics - no mocking
        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            inverse_transform=scaler,
            apply_inverse_scaling=True,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Should handle extreme values correctly
        evaluator.update(data_extreme_values)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # Real metrics should compute both normalized and denormalized
        assert "mse_denormalized" in results
        assert "mse_normalized" in results
        assert results["mse_denormalized"] >= 0.0
        assert results["mse_normalized"] >= 0.0
        # Denormalized should be larger (scaled by 2x, so error scales by 4x)
        assert results["mse_denormalized"] > results["mse_normalized"]


class TestLargeBatchHandling:
    """Tests with large batches (performance/stress testing)."""

    def test_large_batch_regression(self, regression_data_large_batch):
        """Test evaluator handles large batches efficiently."""
        # Use real metrics - no mocking
        evaluator = DefaultEvaluator(
            metric_names=["mse", "rmse"],
            task_type="regression",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Should handle large batch without issues
        evaluator.update(regression_data_large_batch)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert isinstance(results, dict)
        assert "mse_normalized" in results
        assert "rmse_normalized" in results

        # Verify metrics are reasonable with real computations
        assert (
            results["mse_normalized"] >= 0.0
        ), f"MSE should be non-negative, got {results['mse_normalized']}"
        assert (
            results["rmse_normalized"] >= 0.0
        ), f"RMSE should be non-negative, got {results['rmse_normalized']}"
        assert np.isfinite(
            results["mse_normalized"]
        ), f"MSE should be finite, got {results['mse_normalized']}"
        assert np.isfinite(
            results["rmse_normalized"]
        ), f"RMSE should be finite, got {results['rmse_normalized']}"
        # RMSE should be sqrt of MSE (approximately)
        assert (
            abs(results["rmse_normalized"] - np.sqrt(results["mse_normalized"])) < 1e-5
        )

    def test_large_batch_multiunit(self, regression_data_large_batch):
        """Test multi-unit evaluator with large batch."""
        # Use real metrics - no mocking
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Create unit IDs for large batch
        batch_size = len(regression_data_large_batch["predictions"])
        # Distribute across 10 units
        unit_ids = np.random.RandomState(42).randint(1, 11, size=batch_size)

        large_multiunit_data = regression_data_large_batch.copy()
        large_multiunit_data["unit_id"] = unit_ids

        # Should handle efficiently
        evaluator.update(large_multiunit_data)
        # FIX: Use public compute() instead of deleted _compute_metrics()
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert isinstance(results, dict)
        assert "mse_normalized_mean" in results
        assert results["mse_normalized_mean"] >= 0.0
        assert np.isfinite(results["mse_normalized_mean"])


class TestSingleSampleHandling:
    """Tests with single sample (minimum batch size)."""

    def test_single_sample_regression(self, regression_data_single_sample):
        """Test evaluator handles single sample correctly."""
        # Use real metrics - no mocking
        evaluator = DefaultEvaluator(
            metric_names=["mse"],
            task_type="regression",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        evaluator.update(regression_data_single_sample)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert "mse_normalized" in results
        assert (
            results["mse_normalized"] >= 0.0
        ), f"MSE should be non-negative, got {results['mse_normalized']}"
        assert np.isfinite(
            results["mse_normalized"]
        ), f"MSE should be finite, got {results['mse_normalized']}"

    def test_single_sample_multiunit(self, regression_data_single_sample):
        """Test multi-unit evaluator with single sample."""
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        single_multiunit_data = regression_data_single_sample.copy()
        single_multiunit_data["unit_id"] = np.array([1], dtype=np.int32)

        evaluator.update(single_multiunit_data)
        # FIX: Use public compute() instead of deleted _compute_metrics()
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert isinstance(results, dict)
        # FIX: Check unit_managers instead of deleted unit_metrics
        assert 1 in evaluator.unit_managers


class TestFaultOnsetScenarios:
    """Tests with fault onset data (realistic PHM scenarios)."""

    def test_fault_onset_forecasting(self, forecasting_data_fault_onset):
        """Test forecasting evaluator handles fault onset scenarios."""
        # Use real metrics - no mocking
        evaluator = DefaultEvaluator(
            metric_names=["mse", "mae"],
            task_type="forecasting",
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Should handle fault onset data correctly
        evaluator.update(forecasting_data_fault_onset)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert "mse_normalized" in results
        assert "mae_normalized" in results

        # Fault onset should have higher error than perfect predictions
        assert (
            results["mse_normalized"] > 0.0
        ), f"MSE should be positive for fault onset, got {results['mse_normalized']}"
        assert (
            results["mae_normalized"] > 0.0
        ), f"MAE should be positive for fault onset, got {results['mae_normalized']}"
        assert np.isfinite(results["mse_normalized"])
        assert np.isfinite(results["mae_normalized"])


class TestImbalancedClassification:
    """Tests with imbalanced classification data."""

    def test_imbalanced_data_handling(self, classification_data_imbalanced):
        """Test classification evaluator handles imbalanced data."""
        # Use real metrics - no mocking
        evaluator = ClassificationEvaluator(
            num_classes=3,
            metric_names=["accuracy", "f1", "precision", "recall"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        evaluator.update(classification_data_imbalanced)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # All metrics should be computed with real implementations
        assert "accuracy" in results
        assert "f1" in results
        assert "precision" in results
        assert "recall" in results

        # Values should be in valid range (real metrics)
        assert (
            0.0 <= results["accuracy"] <= 1.0
        ), f"Accuracy should be in [0,1], got {results['accuracy']}"
        assert (
            0.0 <= results["f1"] <= 1.0
        ), f"F1 should be in [0,1], got {results['f1']}"
        assert (
            0.0 <= results["precision"] <= 1.0
        ), f"Precision should be in [0,1], got {results['precision']}"
        assert (
            0.0 <= results["recall"] <= 1.0
        ), f"Recall should be in [0,1], got {results['recall']}"
        assert np.isfinite(results["accuracy"])
        assert np.isfinite(results["f1"])

    def test_misclassified_data_handling(self, classification_data_misclassified):
        """Test evaluator handles misclassified data correctly."""
        # Use real metrics - no mocking
        evaluator = ClassificationEvaluator(
            num_classes=3,
            metric_names=["accuracy"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        evaluator.update(classification_data_misclassified)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # Accuracy should reflect misclassifications (real metric computation)
        assert "accuracy" in results
        # With 25% misclassification rate, accuracy should be around 0.75
        assert (
            0.5 <= results["accuracy"] <= 1.0
        ), f"Accuracy should be reasonable, got {results['accuracy']}"
        assert np.isfinite(results["accuracy"])


class TestManyUnitsScenario:
    """Tests with many units (stress test for unit tracking)."""

    def test_many_units_handling(self, multiunit_data_many_units):
        """Test multi-unit evaluator handles many units efficiently."""
        # Use real metrics - no mocking
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        # Should handle many units efficiently
        evaluator.update(multiunit_data_many_units)
        # FIX: Use public compute() instead of deleted _compute_metrics()
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert isinstance(results, dict)
        assert "mse_normalized_mean" in results
        assert (
            results["mse_normalized_mean"] >= 0.0
        ), f"Mean MSE should be non-negative, got {results['mse_normalized_mean']}"
        assert np.isfinite(
            results["mse_normalized_mean"]
        ), f"Mean MSE should be finite, got {results['mse_normalized_mean']}"

        # Verify per-unit metrics exist (if logging enabled)
        unique_units = np.unique(multiunit_data_many_units["unit_id"])
        if evaluator.log_per_unit_metrics:
            for unit_id in unique_units[:5]:  # Check first 5 units
                key = f"mse_normalized_{unit_id}"
                if key in results:
                    assert results[key] >= 0.0
                    assert np.isfinite(results[key])

        # Verify all units have metrics
        for unit_id in unique_units:
            # FIX: Check unit_managers instead of deleted unit_metrics
            assert unit_id in evaluator.unit_managers

    def test_many_units_time_series(self, multiunit_data_time_series):
        """Test many units with time series data."""
        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
            remote_logger=None,
            collect_predictions=True,
        )

        evaluator.update(multiunit_data_time_series)
        # FIX: Use public compute() instead of deleted _compute_metrics()
        results = evaluator.compute(mode="test", epoch=1, step=1)

        assert isinstance(results, dict)
        # Should aggregate correctly across units and time
        assert "mse_normalized_mean" in results
