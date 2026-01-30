"""
Comprehensive edge case tests using fixture data.

This module tests edge cases and boundary conditions using the edge case
fixtures from conftest.py. These tests ensure robust error handling and
graceful degradation.
"""

import pytest
import numpy as np

from lmetk.evaluator.default import DefaultEvaluator
from lmetk.evaluator.classification import ClassificationEvaluator
from lmetk.evaluator.multiunit import MultiUnitEvaluator


# =============================================================================
# === EDGE CASE TESTS ===
# =============================================================================


class TestEmptyDataHandling:
    """Tests handling of empty data (edge cases)."""

    def test_empty_batch_regression(self, data_empty_batch):
        """Test regression evaluator handles empty batch gracefully."""
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

        # Should not raise
        evaluator.update(data_empty_batch)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # Should return metrics (may be zero or NaN for empty data)
        assert isinstance(results, dict)
        # Real metric should handle empty data gracefully
        if "mse_normalized" in results:
            assert np.isfinite(results["mse_normalized"]) or np.isnan(
                results["mse_normalized"]
            )

    def test_empty_batch_classification(self, data_empty_batch):
        """Test classification evaluator handles empty batch gracefully."""
        # Use real metrics - no mocking
        evaluator = ClassificationEvaluator(
            num_classes=3,
            metric_names=["accuracy"],
            inverse_transform=None,
            apply_inverse_scaling=False,
            save_predictions=False,
            paths={},
        )

        # Convert to classification format
        empty_cls_data = {
            "predictions": np.zeros((0, 1, 3), dtype=np.float32),
            "targets": np.zeros((0, 1, 1), dtype=np.int32),
        }

        # Should not raise
        evaluator.update(empty_cls_data)
        results = evaluator.compute(mode="test", epoch=1, step=1)

        # Real metric should handle empty data gracefully
        assert isinstance(results, dict)
        if "accuracy" in results:
            assert np.isfinite(results["accuracy"]) or np.isnan(results["accuracy"])

    def test_empty_batch_multiunit(self, data_empty_batch):
        """Test multi-unit evaluator handles empty batch gracefully."""
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

        # Add empty unit_ids
        empty_multiunit_data = data_empty_batch.copy()
        empty_multiunit_data["unit_id"] = np.array([], dtype=np.int32)

        # Should not raise
        evaluator.update(empty_multiunit_data)
        results = evaluator._compute_metrics()

        # Should return fallback metrics (real metrics handle empty data)
        assert isinstance(results, dict)
        assert "mse_denormalized_mean" in results or "mse_mean" in results
        # Verify metric value is finite or NaN (empty data case)
        mean_key = (
            "mse_denormalized_mean"
            if "mse_denormalized_mean" in results
            else "mse_mean"
        )
        assert np.isfinite(results[mean_key]) or np.isnan(results[mean_key])


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
        results = evaluator._compute_metrics()

        # Mean should be zero with real metrics
        assert (
            results["mse_normalized_mean"] < 1e-6
        ), f"MSE mean should be near zero, got {results['mse_denormalized_mean']}"


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
        from lmetk.transforms.base_transforms.scaler import ConstantScaler

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
        results = evaluator._compute_metrics()

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
        results = evaluator._compute_metrics()

        assert isinstance(results, dict)
        assert 1 in evaluator.unit_metrics


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
        results = evaluator._compute_metrics()

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
                key = f"mse_{unit_id}_denormalized"
                if key in results:
                    assert results[key] >= 0.0
                    assert np.isfinite(results[key])

        # Verify all units have metrics
        unique_units = np.unique(multiunit_data_many_units["unit_id"])
        for unit_id in unique_units:
            assert unit_id in evaluator.unit_metrics

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
        results = evaluator._compute_metrics()

        assert isinstance(results, dict)
        # Should aggregate correctly across units and time
        assert "mse_normalized_mean" in results
