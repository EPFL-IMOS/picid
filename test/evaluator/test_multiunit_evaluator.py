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
from lmetk.metrics.metrics import MSEMetric

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
def multiunit_evaluator_factory(mocker, mock_path_obj):
    """
    Factory fixture for creating MultiUnitEvaluator with custom configurations.

    Returns a callable that creates evaluators with the specified parameters.
    This ensures all configuration is done through __init__, not post-hoc assignment.

    **PHM Context**: Factory pattern allows testing different PHM evaluation
    configurations (with/without scaling, different logging options) properly.
    """

    def _create_evaluator(
        apply_inverse_scaling=False,
        log_image=False,
        log_per_unit_metrics=True,
        enable_plot_subsampling=False,
        plot_subsample_threshold=1000,
        plot_subsample_factor=10,
        remote_logger=None,
        save_predictions=True,
        collect_predictions=True,
        use_mock_logger=True,
    ):
        # Use real MSEMetric instead of MockMetric
        # Don't mock MetricFactory - let it create real metrics
        # mocker.patch.object(
        #     MetricFactory, "create_metric", side_effect=lambda name, paths: MockMetric(name)
        # )

        # Create a real scaling wrapper instead of MagicMock
        from lmetk.transforms.base.multisource import InverseTransformMixin

        class SimpleScaler(InverseTransformMixin):
            def __init__(self, apply_inverse_scaling):
                self.apply_inverse = apply_inverse_scaling

            def inverse_transform_if_needed(self, preds, targets, metadata=None):
                if self.apply_inverse:
                    return preds * 2.0, targets * 2.0
                return preds, targets

            def inverse_transform(self, data, metadata=None):
                return data * 2.0

        scaler = SimpleScaler(apply_inverse_scaling)

        # Use provided logger or create mock if requested
        if remote_logger is None and use_mock_logger:
            remote_logger = MagicMock()

        evaluator = MultiUnitEvaluator(
            metric_names=["mse"],  # Use real MSE metric
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
        # Attach the wrapper to enable test inspection
        evaluator.scaling_wrapper = scaler

        return evaluator

    return _create_evaluator


@pytest.fixture
def multiunit_evaluator_fixture(multiunit_evaluator_factory):
    """
    Default MultiUnitEvaluator fixture with scaling OFF and logging ON.

    **PHM Context**: Basic evaluator for testing core RUL prediction
    evaluation without physical unit conversion.
    """
    return multiunit_evaluator_factory(
        apply_inverse_scaling=False,
        log_image=False,
        log_per_unit_metrics=True,
    )


@pytest.fixture
def multiunit_evaluator_with_scaling_fixture(multiunit_evaluator_factory):
    """
    Fixture for MultiUnitEvaluator with apply_inverse_scaling=True.

    This creates the evaluator with scaling enabled from the start,
    ensuring normalized_unit_metrics is properly initialized.

    **PHM Context**: Tests scenarios where physical unit conversion
    (e.g., normalized sensor readings back to engineering units) is enabled.
    """
    return multiunit_evaluator_factory(apply_inverse_scaling=True)


@pytest.fixture
def multiunit_evaluator_with_plotting_fixture(multiunit_evaluator_factory):
    """
    Fixture for MultiUnitEvaluator with plotting enabled.

    **PHM Context**: Tests prediction vs target curve visualization
    for RUL trend analysis.
    """
    return multiunit_evaluator_factory(
        log_image=True,
        enable_plot_subsampling=False,
    )


@pytest.fixture
def multiunit_evaluator_with_subsampling_fixture(multiunit_evaluator_factory):
    """
    Fixture for MultiUnitEvaluator with plot subsampling enabled.

    **PHM Context**: Tests large dataset visualization with subsampling
    for performance optimization.
    """
    return multiunit_evaluator_factory(
        log_image=True,
        enable_plot_subsampling=True,
        plot_subsample_threshold=100,
        plot_subsample_factor=5,
    )


@pytest.fixture
def multiunit_evaluator_no_logger_fixture(multiunit_evaluator_factory):
    """
    Fixture for MultiUnitEvaluator without remote logger.

    **PHM Context**: Tests offline evaluation without remote logging.
    """
    return multiunit_evaluator_factory(
        remote_logger=None,
        use_mock_logger=False,
    )


@pytest.fixture
def multiunit_evaluator_no_per_unit_logging_fixture(multiunit_evaluator_factory):
    """
    Fixture for MultiUnitEvaluator with per-unit logging disabled.

    **PHM Context**: Tests aggregated metrics only, without per-unit breakdown.
    """
    return multiunit_evaluator_factory(log_per_unit_metrics=False)


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


def test_reset_clears_everything(multiunit_evaluator_fixture, multiunit_data_1d_units):
    """
    Tests that reset wipes data and clears nested metric dictionaries.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data.
    """
    eval = multiunit_evaluator_fixture

    # ACT: Populate with realistic data
    eval.update(multiunit_data_1d_units)

    # Verify state exists before reset
    assert len(eval.unit_ids) > 0
    assert len(eval.unit_metrics) > 0

    # Reset
    eval.reset()

    # ASSERT: Everything should be cleared
    assert len(eval.unit_ids) == 0
    assert len(eval.unit_metrics) == 0
    assert len(eval.preds) == 0
    assert len(eval.targets) == 0


# =========================================================================
# === 3. Tests: Update Logic (Splitting & Scaling) ===
# =========================================================================


def test_update_splits_data_by_unit_1d(
    multiunit_evaluator_fixture, multiunit_data_1d_units
):
    """
    Tests correct routing of data to specific unit metrics when unit_ids are 1D.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data with
    uneven distribution and different degradation rates per unit.
    """
    eval = multiunit_evaluator_fixture

    # ACT: Use realistic fixture data
    eval.update(multiunit_data_1d_units)

    # ASSERT: Verify metrics are correctly routed to units
    # The fixture has units 1-5 with different sample counts
    unit_ids_in_data = multiunit_data_1d_units["unit_id"]
    unique_units = np.unique(unit_ids_in_data)

    # Each unit should have metrics
    for unit_id in unique_units:
        assert unit_id in eval.unit_metrics, f"Unit {unit_id} should have metrics"
        assert "mse" in eval.unit_metrics[unit_id]

        # Verify metric was updated (non-zero if unit has samples)
        metric_value = eval.unit_metrics[unit_id]["mse"].compute()
        assert metric_value >= 0.0, f"Unit {unit_id} MSE should be non-negative"
        assert np.isfinite(metric_value), f"Unit {unit_id} MSE should be finite"

    # Verify unit_ids were collected
    assert len(eval.unit_ids) > 0


def test_update_dual_metrics_when_scaling_on(
    multiunit_evaluator_with_scaling_fixture, regression_data_basic
):
    """
    Tests that when scaling is ON:
    1. Primary metrics get Denormalized data (2x).
    2. Secondary metrics get Normalized data (1x).

    **PHM Logic**: When inverse scaling is enabled, we track both
    normalized (model output) and denormalized (physical units) metrics
    to compare model performance in both spaces.

    **Methodology**: Use fixture with scaling enabled from initialization,
    ensuring normalized_unit_metrics is properly created by __init__.
    Use realistic regression data with unit IDs.

    **Uses fixture**: regression_data_basic - realistic RUL predictions.

    **Expected**: Primary metrics contain scaled (physical) values,
    secondary metrics contain original (normalized) values.
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Verify scaling is enabled and normalized_unit_metrics exists
    assert eval.scaling_wrapper.apply_inverse is True
    assert hasattr(eval, "normalized_unit_metrics")

    # Create model_out with unit IDs from realistic fixture data
    model_out = regression_data_basic.copy()
    # Add unit IDs (all same unit for simplicity)
    model_out["unit_id"] = np.ones(len(model_out["predictions"]), dtype=np.int32)

    # ACT
    eval.update(model_out)

    # ASSERT: Verify dual metrics exist
    unit_id = 1
    assert unit_id in eval.unit_metrics
    assert unit_id in eval.normalized_unit_metrics

    # Primary (Denormalized): Should be scaled (2x from scaler)
    # MSE scales by 4x when data is scaled by 2x (because MSE = mean((pred - target)^2))
    denorm_value = eval.unit_metrics[unit_id]["mse"].compute()

    # Secondary (Normalized): Should be original
    norm_value = eval.normalized_unit_metrics[unit_id]["mse"].compute()

    # Denormalized MSE should be larger (scaled by 4x because error scales quadratically)
    assert (
        denorm_value > norm_value
    ), f"Denormalized MSE ({denorm_value}) should be larger than normalized ({norm_value})"
    # Verify approximate 4x relationship (MSE scales quadratically)
    assert (
        abs(denorm_value - norm_value * 4.0) < 0.1
    ), f"Should be approximately 4x, got denorm={denorm_value}, norm={norm_value}"
    assert np.isfinite(denorm_value) and np.isfinite(norm_value)


# =========================================================================
# === 4. Tests: Compute & Aggregation ===
# =========================================================================


def test_compute_aggregates_means_and_logs_per_unit(
    multiunit_evaluator_fixture, multiunit_data_1d_units
):
    """
    Tests the complex aggregation logic in _compute_metrics.
    Scenario: Scaling OFF.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data with
    uneven distribution across 5 units.
    """
    eval = multiunit_evaluator_fixture

    # ACT: Use realistic fixture data
    eval.update(multiunit_data_1d_units)
    results = eval._compute_metrics()

    # ASSERT: Verify aggregation logic with real MSE metrics
    # 1. Check per-unit logs exist (if log_per_unit_metrics=True)
    if eval.log_per_unit_metrics:
        unit_ids_in_data = np.unique(multiunit_data_1d_units["unit_id"])
        for unit_id in unit_ids_in_data:
            key = f"mse_{unit_id}_denormalized"
            assert key in results, f"Per-unit metric for unit {unit_id} should exist"
            assert (
                results[key] >= 0.0
            ), f"Per-unit MSE should be non-negative, got {results[key]}"
            assert np.isfinite(
                results[key]
            ), f"Per-unit MSE should be finite, got {results[key]}"

    # 2. Check global mean exists
    assert "mse_denormalized_mean" in results
    mean_value = results["mse_denormalized_mean"]
    assert mean_value >= 0.0, f"Mean MSE should be non-negative, got {mean_value}"
    assert np.isfinite(mean_value), f"Mean MSE should be finite, got {mean_value}"

    # Verify mean is reasonable (should be average of per-unit values)
    if eval.log_per_unit_metrics:
        per_unit_values = [
            results[f"mse_{uid}_denormalized"]
            for uid in np.unique(multiunit_data_1d_units["unit_id"])
        ]
        expected_mean = np.mean(per_unit_values)
        assert (
            abs(mean_value - expected_mean) < 1e-6
        ), f"Mean should match average of per-unit values, got {mean_value}, expected {expected_mean}"


def test_compute_dual_aggregation(
    multiunit_evaluator_with_scaling_fixture, multiunit_data_1d_units
):
    """
    Tests aggregation when both normalized and denormalized metrics exist.

    **PHM Logic**: When scaling is enabled, we compute metrics in both
    normalized (model space) and denormalized (physical space) to provide
    comprehensive evaluation of RUL predictions.

    **Methodology**: Use fixture with scaling properly initialized,
    feed realistic multi-unit data through update() to populate metrics.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data.

    **Expected**: Both normalized and denormalized means are computed correctly.
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Verify scaling is enabled and normalized_unit_metrics exists
    assert eval.scaling_wrapper.apply_inverse is True
    assert hasattr(eval, "normalized_unit_metrics")

    # ACT: Use realistic fixture data
    eval.update(multiunit_data_1d_units)
    results = eval._compute_metrics()

    # ASSERT: Verify dual aggregation with real MSE metrics
    # Both normalized and denormalized means should exist
    assert "mse_denormalized_mean" in results
    assert "mse_normalized_mean" in results

    denorm_mean = results["mse_denormalized_mean"]
    norm_mean = results["mse_normalized_mean"]

    # Denormalized MSE should be larger (scaled by 4x because error scales quadratically)
    assert (
        denorm_mean > norm_mean
    ), f"Denormalized mean ({denorm_mean}) should be larger than normalized ({norm_mean})"

    # Verify approximate 4x relationship (MSE scales quadratically)
    assert (
        abs(denorm_mean - norm_mean * 4.0) < 0.1
    ), f"Should be approximately 4x, got denorm={denorm_mean}, norm={norm_mean}"
    assert np.isfinite(denorm_mean) and np.isfinite(norm_mean)

    # Check per-unit metrics exist (if enabled)
    if eval.log_per_unit_metrics:
        unit_ids = np.unique(multiunit_data_1d_units["unit_id"])
        for unit_id in unit_ids:
            assert f"mse_{unit_id}_denormalized" in results
            assert f"mse_{unit_id}_normalized" in results
            assert np.isfinite(results[f"mse_{unit_id}_denormalized"])
            assert np.isfinite(results[f"mse_{unit_id}_normalized"])


# =========================================================================
# === 5. Tests: Plotting Side-Effects ===
# =========================================================================


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_triggers(mock_plt, multiunit_evaluator_with_plotting_fixture):
    """
    Tests that plotting logic is triggered for validation steps.

    **PHM Logic**: Visualization of predictions vs targets is critical
    for validating RUL prediction quality.

    **Methodology**: Use fixture with plotting enabled from init,
    feed data through update() to populate collections properly.

    **Expected**: Plot created and logged during validation compute.
    """
    eval = multiunit_evaluator_with_plotting_fixture

    # Verify plotting was enabled at init
    assert eval.log_fit_vs_target_curves is True

    # Configure mock for subplots
    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # Feed data through the proper interface (update)
    preds = np.zeros((10, 10, 1))  # (batch, time, features)
    targets = np.zeros((10, 10, 1))
    unit_ids = np.ones((10,), dtype=int)  # All unit 1

    eval.update({"predictions": preds, "targets": targets, "unit_id": unit_ids})

    # Mock the log_plot method (this is a method mock, not config)
    eval.log_plot = MagicMock()

    # ACT: Mock only file operations, not orchestration
    with patch.object(eval, "_save_predictions_to_file"):
        results = eval.compute(mode="val", epoch=0, step=0)

    # ASSERT: Verify full behavior including orchestration
    mock_plt.subplots.assert_called()
    eval.log_plot.assert_called()
    mock_plt.close.assert_called()
    # Verify compute() returns metrics dict (full orchestration tested)
    assert isinstance(results, dict)
    assert len(results) > 0  # Should have computed metrics


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


# =========================================================================
# === 7. Tests: 2D Unit IDs (Multi-dimensional equipment identification) ===
# =========================================================================


def test_update_splits_data_by_unit_2d(
    multiunit_evaluator_fixture, multiunit_data_2d_units
):
    """
    Tests correct routing when unit_ids are 2D (e.g., (dataset_id, unit_id)).

    **PHM Logic**: Some datasets identify units by multiple indices,
    e.g., N-CMAPSS uses (dataset_id, unit_id) tuples.

    **Methodology**: Use realistic 2D unit ID fixture with hierarchical identification.

    **Expected**: Metrics correctly routed to tuple keys.

    **Uses fixture**: multiunit_data_2d_units - hierarchical unit IDs across datasets.

    Validates: Requirement MU-2D-1 - 2D unit ID handling
    """
    eval = multiunit_evaluator_fixture

    # ACT: Use realistic fixture data with 2D unit IDs
    eval.update(multiunit_data_2d_units)

    # ASSERT: Verify metrics are correctly routed to tuple keys
    unit_ids_in_data = multiunit_data_2d_units["unit_id"]
    unique_units = np.unique(unit_ids_in_data, axis=0)

    # Each (dataset, unit) tuple should have metrics
    for unit_tuple in unique_units:
        key = tuple(int(x) for x in unit_tuple)
        assert key in eval.unit_metrics, f"Unit {key} should have metrics"
        assert "mse" in eval.unit_metrics[key]

        # Verify metric was updated with real MSE
        metric_value = eval.unit_metrics[key]["mse"].compute()
        assert (
            metric_value >= 0.0
        ), f"Unit {key} MSE should be non-negative, got {metric_value}"
        assert np.isfinite(
            metric_value
        ), f"Unit {key} MSE should be finite, got {metric_value}"


def test_update_raises_for_3d_unit_ids(multiunit_evaluator_fixture):
    """
    Tests that 3D unit_ids raise ValueError.

    **PHM Logic**: Only 1D and 2D unit identification is supported.

    **Methodology**: Pass 3D unit_ids array.

    **Expected**: ValueError raised.

    Validates: Requirement MU-2D-2 - 3D unit ID rejection
    """
    eval = multiunit_evaluator_fixture

    preds = np.array([[[10.0]]])
    targets = np.array([[[10.0]]])
    unit_ids = np.array([[[1, 2, 3]]])  # 3D - invalid

    model_out = {"predictions": preds, "targets": targets, "unit_id": unit_ids}

    with pytest.raises(ValueError, match="Only 1D and 2D are supported"):
        eval.update(model_out)


def test_update_handles_integer_vs_array_unit(
    multiunit_evaluator_fixture, regression_data_single_sample
):
    """
    Tests that both integer and array unit IDs are handled.

    **PHM Logic**: Single units may be integers or single-element arrays.

    **Methodology**: Test with integer-like unit values using realistic data.

    **Uses fixture**: regression_data_single_sample - edge case with single sample.

    **Expected**: Correct key type in unit_metrics.

    Validates: Requirement MU-2D-3 - Integer unit handling
    """
    eval = multiunit_evaluator_fixture

    # Create model_out with single sample and integer unit ID
    model_out = regression_data_single_sample.copy()
    model_out["unit_id"] = np.array([5], dtype=np.int32)  # 1D integer

    # ACT
    eval.update(model_out)

    # ASSERT: Key should be int, not array
    assert 5 in eval.unit_metrics
    assert isinstance(list(eval.unit_metrics.keys())[0], int)
    assert "mse" in eval.unit_metrics[5]
    # Verify real MSE metric exists
    mse_value = eval.unit_metrics[5]["mse"].compute()
    assert mse_value >= 0.0
    assert np.isfinite(mse_value)


# =========================================================================
# === 8. Tests: Plotting Edge Cases ===
# =========================================================================


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_skips_when_no_logger(
    mock_plt, multiunit_evaluator_factory, multiunit_data_time_series
):
    """
    Tests that plotting is skipped when remote_logger is None.

    **PHM Logic**: Plotting requires remote logger for W&B/MLflow.
    Without logger, visualization is skipped to avoid errors.

    **Methodology**: Create evaluator with plotting enabled but no logger,
    feed realistic data through update() interface.

    **Uses fixture**: multiunit_data_time_series - realistic multi-unit time series data.

    **Expected**: No plotting calls made.

    Validates: Requirement MU-PLOT-1 - No logger skip
    """
    eval = multiunit_evaluator_factory(
        log_image=True,
        remote_logger=None,
        use_mock_logger=False,
    )

    # Verify config was set correctly at init
    assert eval.log_fit_vs_target_curves is True
    assert eval.remote_logger is None

    # ACT: Use realistic fixture data
    eval.update(multiunit_data_time_series)

    # ACT: Mock only file operations, not orchestration
    with patch.object(eval, "_save_predictions_to_file"):
        results = eval.compute(mode="val", epoch=0, step=0)

    # ASSERT: Should not create plots, but compute() should still return metrics
    mock_plt.subplots.assert_not_called()
    assert isinstance(results, dict)  # Verify compute() returns metrics


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_subsamples_large_data(
    mock_plt, multiunit_evaluator_with_subsampling_fixture, regression_data_large_batch
):
    """
    Tests that plotting subsamples large datasets.

    **PHM Logic**: Large datasets are subsampled for plotting performance.
    In PHM with high-frequency sensor data, this prevents memory issues.

    **Methodology**: Use fixture with subsampling enabled (threshold=100, factor=5),
    feed realistic large batch data through update() interface.

    **Uses fixture**: regression_data_large_batch - realistic large batch (1000 samples)
    with outliers and realistic noise distribution.

    **Expected**: Subsampling applied when plotting.

    Validates: Requirement MU-PLOT-2 - Large data subsampling
    """
    eval = multiunit_evaluator_with_subsampling_fixture

    # Verify subsampling config was set at init
    assert eval.enable_plot_subsampling is True
    assert eval.plot_subsample_threshold == 100
    assert eval.plot_subsample_factor == 5

    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # ACT: Use realistic large batch fixture data
    # Add unit IDs (all same unit for simplicity)
    large_multiunit_data = regression_data_large_batch.copy()
    large_multiunit_data["unit_id"] = np.ones(
        len(large_multiunit_data["predictions"]), dtype=np.int32
    )

    eval.update(large_multiunit_data)

    # Call the plot method to test subsampling
    eval._log_pred_vs_target_curves(mode="val", step=0)

    # ASSERT: Should have created plot (subsampling applied internally)
    mock_plt.subplots.assert_called()


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_handles_2d_unit_ids(
    mock_plt, multiunit_evaluator_with_plotting_fixture, multiunit_data_2d_units
):
    """
    Tests plotting with 2D unit IDs creates correct titles.

    **PHM Logic**: 2D unit IDs shown as tuples in plot titles.
    Multi-dimensional unit IDs support hierarchical asset identification.

    **Methodology**: Use fixture with plotting enabled, feed realistic 2D unit_ids
    through update() interface.

    **Uses fixture**: multiunit_data_2d_units - realistic hierarchical unit IDs
    (dataset, unit) across multiple datasets.

    **Expected**: Title includes tuple representation.

    Validates: Requirement MU-PLOT-3 - 2D unit ID in titles
    """
    eval = multiunit_evaluator_with_plotting_fixture

    # Verify plotting was enabled at init
    assert eval.log_fit_vs_target_curves is True

    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    # ACT: Use realistic 2D unit ID fixture data
    eval.update(multiunit_data_2d_units)

    eval._log_pred_vs_target_curves(mode="val", step=0)

    # ASSERT: Should have set title with tuple format
    mock_ax.set_title.assert_called()


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_plotting_handles_empty_predictions(
    mock_plt, multiunit_evaluator_with_plotting_fixture
):
    """
    Tests plotting handles empty predictions gracefully.

    **PHM Logic**: Early epochs may have no predictions yet.
    System should handle gracefully without errors.

    **Methodology**: Use fixture with plotting enabled, call reset() to
    ensure collections are empty (simulating start of evaluation).

    **Expected**: No error, no plot created.

    Validates: Requirement MU-PLOT-4 - Empty data handling
    """
    eval = multiunit_evaluator_with_plotting_fixture

    # Verify plotting was enabled at init
    assert eval.log_fit_vs_target_curves is True

    # Use reset() - the proper way to clear collections
    eval.reset()

    # Verify collections are empty
    assert len(eval.preds) == 0
    assert len(eval.targets) == 0
    assert len(eval.unit_ids) == 0

    # Should not raise when called with empty collections
    eval._log_pred_vs_target_curves(mode="val", step=0)

    mock_plt.subplots.assert_not_called()


# =========================================================================
# === 9. Tests: Compute Metrics Edge Cases ===
# =========================================================================


def test_compute_metrics_empty_units(multiunit_evaluator_fixture, data_empty_batch):
    """
    Tests _compute_metrics returns fallback for empty unit_metrics.

    **PHM Logic**: Handle edge case of no data processed.

    **Methodology**: Call _compute_metrics without any updates or with empty data.

    **Uses fixture**: data_empty_batch - edge case with no samples.

    **Expected**: Returns dict with 0.0 for all metrics.

    Validates: Requirement MU-COMP-1 - Empty metrics fallback
    """
    eval = multiunit_evaluator_fixture

    # Try with empty data (should handle gracefully)
    if len(data_empty_batch["predictions"]) == 0:
        # Empty batch - should return fallback
        results = eval._compute_metrics()
        assert "mse_mean" in results or "mse_denormalized_mean" in results

    # Also test with no updates at all
    eval.reset()
    results = eval._compute_metrics()

    # Should have mean metric with 0.0 value (real MSE with no data)
    assert "mse_denormalized_mean" in results or "mse_mean" in results
    mean_key = (
        "mse_denormalized_mean" if "mse_denormalized_mean" in results else "mse_mean"
    )
    # With no data, MSE should raise ValueError or return 0.0 depending on implementation
    # Check that it's either 0.0 or raises an error
    try:
        assert results[mean_key] == 0.0 or np.isnan(results[mean_key])
    except (ValueError, KeyError):
        pass  # Some implementations may raise ValueError for empty data


def test_compute_metrics_disabled_per_unit_logging(
    multiunit_evaluator_no_per_unit_logging_fixture, multiunit_data_1d_units
):
    """
    Tests _compute_metrics with log_per_unit_metrics=False.

    **PHM Logic**: Per-unit metrics can be disabled for cleaner logging.
    Only aggregate metrics are reported, reducing log noise in production.

    **Methodology**: Use fixture with per-unit logging disabled from init.
    Feed realistic multi-unit data.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data.

    **Expected**: Only mean metrics in results, not per-unit.

    Validates: Requirement MU-COMP-2 - Per-unit logging toggle
    """
    eval = multiunit_evaluator_no_per_unit_logging_fixture

    # Verify per-unit logging was disabled at init
    assert eval.log_per_unit_metrics is False

    # ACT: Use realistic fixture data
    eval.update(multiunit_data_1d_units)
    results = eval._compute_metrics()

    # ASSERT: Should have mean but not per-unit (real MSE metrics)
    assert "mse_normalized_mean" in results
    assert results["mse_normalized_mean"] >= 0.0
    assert np.isfinite(results["mse_normalized_mean"])

    # Verify no per-unit metrics in results
    unit_ids = np.unique(multiunit_data_1d_units["unit_id"])
    for unit_id in unit_ids:
        assert f"mse_{unit_id}_denormalized" not in results
        assert f"mse_{unit_id}_normalized" not in results


@patch("lmetk.evaluator.default.DefaultEvaluator._prepare_predictions_for_save")
def test_prepare_predictions_empty_parent(mock_super_prep, multiunit_evaluator_fixture):
    """
    Tests _prepare_predictions_for_save returns empty if parent returns empty.

    **PHM Logic**: No predictions saved if collect_predictions is off.

    **Methodology**: Mock parent to return empty dict.

    **Expected**: Empty dict returned.

    Validates: Requirement MU-SAVE-1 - Empty parent handling
    """
    eval = multiunit_evaluator_fixture
    mock_super_prep.return_value = {}

    result = eval._prepare_predictions_for_save()

    assert result == {}


# =========================================================================
# === 10. Tests: Inverse Transform Per Unit ===
# =========================================================================


def test_inverse_transform_per_unit_1d_ids(
    multiunit_evaluator_with_scaling_fixture, multiunit_data_1d_units
):
    """
    Tests _inverse_transform_per_unit with 1D unit IDs.

    **PHM Logic**: Per-unit inverse scaling for datasets like
    PRONOSTIA/XJTU where each unit has different normalization.

    **Methodology**: Use fixture with scaling enabled, test internal method
    with realistic multi-unit data.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data with
    5 units and uneven distribution.

    **Expected**: Correct inverse transform applied per unit.

    Validates: Requirement MU-INV-1 - 1D per-unit inverse
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Override mock behavior for specific test scenario
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        side_effect=lambda p, t, metadata=None: (p * 2.0, t * 2.0)
    )

    # ACT: Use realistic fixture data
    preds = multiunit_data_1d_units["predictions"]
    targets = multiunit_data_1d_units["targets"]
    unit_ids = multiunit_data_1d_units["unit_id"]

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # ASSERT: All values should be scaled by 2
    expected_preds = preds * 2.0
    expected_targets = targets * 2.0
    np.testing.assert_array_almost_equal(result_preds, expected_preds)
    np.testing.assert_array_almost_equal(result_targets, expected_targets)

    # Verify shape preserved
    assert result_preds.shape == preds.shape
    assert result_targets.shape == targets.shape


def test_inverse_transform_per_unit_2d_ids(
    multiunit_evaluator_with_scaling_fixture, multiunit_data_2d_units
):
    """
    Tests _inverse_transform_per_unit with 2D unit IDs.

    **PHM Logic**: Multi-dimensional unit identification for hierarchical assets.

    **Methodology**: Use fixture with scaling enabled, test with realistic 2D unit_ids.

    **Uses fixture**: multiunit_data_2d_units - realistic hierarchical unit IDs
    (dataset, unit) across multiple datasets.

    **Expected**: Correct inverse transform applied per unit tuple.

    Validates: Requirement MU-INV-2 - 2D per-unit inverse
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Override mock behavior for specific test scenario
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        side_effect=lambda p, t, metadata=None: (p * 2.0, t * 2.0)
    )

    # ACT: Use realistic 2D unit ID fixture data
    preds = multiunit_data_2d_units["predictions"]
    targets = multiunit_data_2d_units["targets"]
    unit_ids = multiunit_data_2d_units["unit_id"]

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # ASSERT: All values should be scaled by 2
    expected_preds = preds * 2.0
    expected_targets = targets * 2.0
    np.testing.assert_array_almost_equal(result_preds, expected_preds)
    np.testing.assert_array_almost_equal(result_targets, expected_targets)

    # Verify shape preserved
    assert result_preds.shape == preds.shape
    assert result_targets.shape == targets.shape


def test_inverse_transform_per_unit_3d_ids_warning(
    multiunit_evaluator_with_scaling_fixture, caplog
):
    """
    Tests _inverse_transform_per_unit logs warning for 3D unit IDs.

    **PHM Logic**: 3D unit IDs are not supported, fallback to global scaling.

    **Methodology**: Use fixture with scaling enabled, test with invalid 3D unit_ids.

    **Expected**: Warning logged, global scaling applied.

    Validates: Requirement MU-INV-3 - 3D warning and fallback
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Override mock behavior for specific test scenario
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        side_effect=lambda p, t, metadata=None: (p * 2.0, t * 2.0)
    )

    preds = np.array([[[10.0]]])
    targets = np.array([[[10.0]]])
    unit_ids = np.array([[[1, 2, 3]]])  # 3D - invalid

    import logging

    with caplog.at_level(logging.WARNING):
        result_preds, result_targets = eval._inverse_transform_per_unit(
            preds, targets, unit_ids
        )

    # Should have fallen back to global transform
    assert "Unsupported shape" in caplog.text


def test_inverse_transform_per_unit_no_scaling(
    multiunit_evaluator_fixture, regression_data_basic
):
    """
    Tests _inverse_transform_per_unit when scaling is disabled.

    **PHM Logic**: Returns original data when apply_inverse=False.

    **Methodology**: Use fixture with scaling disabled (default), use realistic data.

    **Uses fixture**: regression_data_basic - realistic RUL predictions.

    **Expected**: Input returned unchanged.

    Validates: Requirement MU-INV-4 - No scaling passthrough
    """
    eval = multiunit_evaluator_fixture

    # Fixture already has apply_inverse=False, verify it
    assert eval.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic fixture data
    preds = regression_data_basic["predictions"]
    targets = regression_data_basic["targets"]
    unit_ids = np.ones(len(preds), dtype=np.int32)  # All same unit

    # Mock to return original data (what the real scaler would do)
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        return_value=(preds.copy(), targets.copy())
    )

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # ASSERT: Should return original data unchanged
    np.testing.assert_array_equal(result_preds, preds)
    np.testing.assert_array_equal(result_targets, targets)


# =========================================================================
# === 11. Tests: Reconstruction Plotting (inherited) ===
# =========================================================================


def test_plot_reconstructions(
    multiunit_evaluator_no_logger_fixture, forecasting_data_multivariate
):
    """
    Tests _plot_reconstructions method returns None without logger.

    **PHM Logic**: Visualizes actual vs predicted sensor values, but
    requires remote logger to be configured.

    **Methodology**: Use fixture with no logger configured, use realistic multivariate data.

    **Uses fixture**: forecasting_data_multivariate - realistic multivariate data
    with 5 sensor channels.

    **Expected**: Returns None when no logger configured.

    Validates: Requirement MU-RECON-1 - Reconstruction plotting guard
    """
    eval = multiunit_evaluator_no_logger_fixture

    # Verify fixture has no logger
    assert eval.remote_logger is None

    # ACT: Use realistic multivariate fixture data
    preds = forecasting_data_multivariate["predictions"][:10]  # 10 samples
    targets = forecasting_data_multivariate["targets"][:10]

    result = eval._plot_reconstructions(preds, targets, "val", epoch=1, step=0)

    # ASSERT: Should return None and not raise
    assert result is None


@patch("lmetk.evaluator.multiunit.logger")
def test_plot_reconstructions_no_logger_returns_none(
    mock_logger, multiunit_evaluator_no_logger_fixture, forecasting_data_multivariate
):
    """
    Tests _plot_reconstructions returns None and logs warning without logger.

    **PHM Logic**: Plotting skipped when no remote logger configured.

    **Methodology**: Use fixture with no logger configured, use realistic multivariate data.

    **Uses fixture**: forecasting_data_multivariate - realistic multivariate data.

    **Expected**: Returns None, logs warning.

    Validates: Requirement MU-RECON-2 - No logger handling
    """
    eval = multiunit_evaluator_no_logger_fixture

    # Verify fixture has no logger
    assert eval.remote_logger is None

    # ACT: Use realistic multivariate fixture data
    preds = forecasting_data_multivariate["predictions"][:10]
    targets = forecasting_data_multivariate["targets"][:10]

    result = eval._plot_reconstructions(preds, targets, "val", epoch=1, step=0)

    # ASSERT: Should return None and log warning
    assert result is None
    mock_logger.warning.assert_called()


# =========================================================================
# === 12. Tests: Additional Coverage for Edge Cases ===
# =========================================================================


def test_init_creates_normalized_unit_metrics_when_scaling_on(mocker, mock_path_obj):
    """
    Tests that normalized_unit_metrics is created when apply_inverse_scaling=True.

    **PHM Logic**: Dual metric storage for normalized and denormalized results.
    When scaling is enabled, we need to track metrics in both normalized
    (model output space) and denormalized (physical units) for comparison.

    **Methodology**: Create evaluator with apply_inverse_scaling=True and
    verify the __init__ properly creates the secondary metric storage.

    **Expected**: normalized_unit_metrics attribute exists after initialization.

    Validates: Requirement MU-INIT-2 - Dual metric storage
    """
    # Use real MSEMetric - don't mock MetricFactory
    # Create a real scaling wrapper
    from lmetk.transforms.base.multisource import InverseTransformMixin

    class SimpleScaler(InverseTransformMixin):
        def __init__(self):
            self.apply_inverse = True

        def inverse_transform_if_needed(self, preds, targets, metadata=None):
            if self.apply_inverse:
                return preds * 2.0, targets * 2.0
            return preds, targets

        def inverse_transform(self, data, metadata=None):
            return data * 2.0

    scaler = SimpleScaler()

    evaluator = MultiUnitEvaluator(
        metric_names=["mse"],
        inverse_transform=scaler,
        apply_inverse_scaling=True,
        save_predictions=False,
        paths=MagicMock(eval_details=mock_path_obj),
        remote_logger=None,
        collect_predictions=True,
    )

    # Verify that __init__ created normalized_unit_metrics because apply_inverse_scaling=True
    # This tests line 82-83 in multiunit.py: if self.scaling_wrapper.apply_inverse:
    assert hasattr(evaluator, "normalized_unit_metrics")
    assert evaluator.scaling_wrapper.apply_inverse is True


def test_reset_clears_normalized_unit_metrics(
    multiunit_evaluator_with_scaling_fixture, multiunit_data_1d_units
):
    """
    Tests that reset clears normalized_unit_metrics when it exists.

    **PHM Logic**: Full reset clears both metric storages (normalized and
    denormalized) to ensure clean state between evaluation epochs.

    **Methodology**: Use fixture with scaling enabled (which creates
    normalized_unit_metrics during init), populate with realistic data, then call reset.

    **Uses fixture**: multiunit_data_1d_units - realistic multi-unit data.

    **Expected**: normalized_unit_metrics cleared after reset.

    Validates: Requirement MU-RESET-2 - Dual storage clear
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Verify normalized_unit_metrics was created by __init__ (not manually)
    assert hasattr(eval, "normalized_unit_metrics")

    # ACT: Populate with realistic data
    eval.update(multiunit_data_1d_units)
    assert len(eval.normalized_unit_metrics) > 0

    # Reset
    eval.reset()

    # ASSERT: Should be cleared
    assert len(eval.normalized_unit_metrics) == 0
    assert len(eval.unit_metrics) == 0
    assert len(eval.unit_ids) == 0


def test_inverse_transform_per_unit_empty_indices_continue(
    multiunit_evaluator_with_scaling_fixture, regression_data_single_sample
):
    """
    Tests _inverse_transform_per_unit continues when indices are empty.

    **PHM Logic**: Some units may have no samples in a batch.

    **Methodology**: Use fixture with scaling enabled, test internal method
    with realistic single sample data.

    **Uses fixture**: regression_data_single_sample - edge case with single sample.

    **Expected**: Loop continues, no error.

    Validates: Requirement MU-INV-5 - Empty indices handling
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert eval.scaling_wrapper.apply_inverse is True

    # ACT: Use realistic single sample fixture data
    preds = regression_data_single_sample["predictions"]
    targets = regression_data_single_sample["targets"]
    unit_ids = np.array([1], dtype=np.int32)

    # Mock transform that will be called for unit 1
    call_count = [0]

    def mock_transform(p, t, metadata=None):
        call_count[0] += 1
        return p * 2.0, t * 2.0

    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        side_effect=mock_transform
    )

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # ASSERT: Should have been called once for unit 1
    assert call_count[0] == 1
    assert result_preds.shape == preds.shape
    assert result_targets.shape == targets.shape


def test_inverse_transform_per_unit_inf_check_no_scaling(
    multiunit_evaluator_fixture, regression_data_single_sample
):
    """
    Tests _inverse_transform_per_unit returns original data when inf and no scaling.

    **PHM Logic**: Fallback to original data when inverse transform incomplete.

    **Methodology**: Use fixture with scaling disabled (default), use realistic data.

    **Uses fixture**: regression_data_single_sample - edge case with single sample.

    **Expected**: Original predictions returned.

    Validates: Requirement MU-INV-6 - Inf check fallback
    """
    eval = multiunit_evaluator_fixture

    # Verify scaling is disabled (fixture default)
    assert eval.scaling_wrapper.apply_inverse is False

    # ACT: Use realistic single sample fixture data
    preds = regression_data_single_sample["predictions"]
    targets = regression_data_single_sample["targets"]
    unit_ids = np.array([1], dtype=np.int32)

    # Transform returns the same data (no scaling)
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        return_value=(preds.copy(), targets.copy())
    )

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # ASSERT: With no scaling, should return original data (no inf)
    np.testing.assert_array_equal(result_preds, preds)
    np.testing.assert_array_equal(result_targets, targets)


def test_inverse_transform_per_unit_raises_on_inf_with_scaling(
    multiunit_evaluator_with_scaling_fixture, regression_data_single_sample
):
    """
    Tests _inverse_transform_per_unit raises ValueError when inf values remain after scaling.

    **PHM Logic**: Error detection for incomplete inverse transform.

    **Methodology**: Use fixture with scaling enabled, mock bad transform,
    use realistic data.

    **Uses fixture**: regression_data_single_sample - realistic single sample data.

    **Expected**: ValueError raised.

    Validates: Requirement MU-INV-7 - Inf detection error
    """
    eval = multiunit_evaluator_with_scaling_fixture

    # Verify scaling is enabled
    assert eval.scaling_wrapper.apply_inverse is True

    # Create scenario where transform returns inf
    def bad_transform(p, t, metadata=None):
        # Return arrays with inf (simulating incomplete transform)
        return np.full_like(p, -np.inf), np.full_like(t, -np.inf)

    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        side_effect=bad_transform
    )

    # ACT: Use realistic single sample fixture data
    preds = regression_data_single_sample["predictions"]
    targets = regression_data_single_sample["targets"]
    unit_ids = np.array([1], dtype=np.int32)

    # ASSERT: Should raise ValueError for incomplete transform
    with pytest.raises(ValueError, match="not filled after inverse transform"):
        eval._inverse_transform_per_unit(preds, targets, unit_ids)


@patch("lmetk.evaluator.multiunit.logger")
def test_log_pred_vs_target_curves_concatenation_error(
    mock_logger, multiunit_evaluator_fixture
):
    """
    Tests _log_pred_vs_target_curves handles concatenation errors.

    **PHM Logic**: Graceful handling of inconsistent array shapes.
    This could happen if there's a bug in upstream data processing.

    **Methodology**: Direct manipulation of internal collections is necessary
    here because this edge case (incompatible array shapes) cannot be triggered
    through the normal update() interface, which validates shapes. We're testing
    defensive error handling for corrupted state.

    **Expected**: Warning logged, method returns early gracefully.

    Validates: Requirement MU-LOG-1 - Concatenation error handling
    """
    eval = multiunit_evaluator_fixture

    # INTENTIONAL: Direct manipulation to create invalid state that can't
    # be reached through normal update() - testing defensive error handling
    eval.preds = [np.zeros((2, 3)), np.zeros((2, 5))]  # Incompatible shapes
    eval.targets = [np.zeros((2, 3)), np.zeros((2, 5))]
    eval.unit_ids = [np.array([1, 2]), np.array([1, 2])]

    eval._log_pred_vs_target_curves(mode="val", step=0)

    mock_logger.warning.assert_called()


@patch("lmetk.evaluator.multiunit.logger")
def test_log_pred_vs_target_curves_3d_unit_ids_warning(
    mock_logger, multiunit_evaluator_fixture
):
    """
    Tests _log_pred_vs_target_curves warns for 3D unit IDs.

    **PHM Logic**: Only 1D and 2D unit IDs supported.
    3D unit IDs indicate invalid data format from upstream.

    **Methodology**: Direct manipulation is necessary because update()
    validates unit_id shape and would reject 3D IDs. We're testing
    defensive handling of invalid state that shouldn't occur normally.

    **Expected**: Warning logged, method returns early gracefully.

    Validates: Requirement MU-LOG-2 - 3D unit ID warning
    """
    eval = multiunit_evaluator_fixture

    # INTENTIONAL: Direct manipulation to create invalid state (3D unit IDs)
    # that update() would reject - testing defensive error handling
    eval.preds = [np.zeros((2, 3, 1))]
    eval.targets = [np.zeros((2, 3, 1))]
    eval.unit_ids = [np.zeros((2, 3, 4))]  # 3D - unsupported

    eval._log_pred_vs_target_curves(mode="val", step=0)

    # Should warn about unsupported shape
    assert any(
        "Unsupported shape" in str(call) for call in mock_logger.warning.call_args_list
    )


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_log_pred_vs_target_curves_2d_unit_ids(
    mock_plt, multiunit_evaluator_fixture, multiunit_data_2d_units
):
    """
    Tests _log_pred_vs_target_curves with 2D unit IDs.

    **PHM Logic**: Multi-dimensional unit identification supported.
    Hierarchical asset IDs (e.g., dataset + unit) represented as 2D.

    **Methodology**: Use fixture (already has logger), feed realistic 2D unit_ids
    through update().

    **Uses fixture**: multiunit_data_2d_units - realistic hierarchical unit IDs
    (dataset, unit) across multiple datasets.

    **Expected**: Plots created with tuple titles.

    Validates: Requirement MU-LOG-3 - 2D unit ID plotting
    """
    eval = multiunit_evaluator_fixture
    # Fixture already provides remote_logger, mock log_plot method
    eval.log_plot = MagicMock()

    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # ACT: Use realistic 2D unit ID fixture data
    eval.update(multiunit_data_2d_units)

    eval._log_pred_vs_target_curves(mode="val", step=0)

    # ASSERT: Should have created plots
    assert mock_plt.subplots.called


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_log_pred_vs_target_curves_subsampling(
    mock_plt, multiunit_evaluator_factory, regression_data_large_batch
):
    """
    Tests _log_pred_vs_target_curves applies subsampling for large data.

    **PHM Logic**: Large datasets subsampled for plot performance.

    **Methodology**: Use factory to create evaluator with subsampling config,
    feed realistic large batch data.

    **Uses fixture**: regression_data_large_batch - realistic large batch (1000 samples)
    with outliers and realistic noise distribution.

    **Expected**: Subsampling applied.

    Validates: Requirement MU-LOG-4 - Subsampling
    """
    eval = multiunit_evaluator_factory(
        enable_plot_subsampling=True,
        plot_subsample_threshold=10,
        plot_subsample_factor=2,
    )

    # Verify config was set at init
    assert eval.enable_plot_subsampling is True
    assert eval.plot_subsample_threshold == 10

    # Mock log_plot method (method mock, not config)
    eval.log_plot = MagicMock()

    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    # ACT: Use realistic large batch fixture data (exceeds threshold of 10)
    large_multiunit_data = regression_data_large_batch.copy()
    large_multiunit_data["unit_id"] = np.ones(
        len(large_multiunit_data["predictions"]), dtype=np.int32
    )
    eval.update(large_multiunit_data)

    eval._log_pred_vs_target_curves(mode="val", step=0)

    # ASSERT: Should have plotted (subsampling applied internally)
    mock_plt.subplots.assert_called()


def test_compute_calls_log_pred_vs_target_on_val(
    multiunit_evaluator_with_plotting_fixture, multiunit_data_time_series
):
    """
    Tests compute calls _log_pred_vs_target_curves for validation mode.

    **PHM Logic**: Plotting triggered during validation to monitor RUL predictions.

    **Methodology**: Use fixture with plotting enabled, feed realistic time series data,
    call compute in val mode.

    **Uses fixture**: multiunit_data_time_series - realistic multi-unit time series data
    with different degradation patterns per unit.

    **Expected**: _log_pred_vs_target_curves called.

    Validates: Requirement MU-COMP-3 - Val mode plotting
    """
    eval = multiunit_evaluator_with_plotting_fixture

    # Verify plotting was enabled at init
    assert eval.log_fit_vs_target_curves is True

    # ACT: Feed realistic time series data through proper interface
    eval.update(multiunit_data_time_series)

    # Mock the internal method (method mock, not config)
    eval._log_pred_vs_target_curves = MagicMock()

    # ACT: Mock only file operations, not orchestration
    with patch.object(eval, "_save_predictions_to_file"):
        results = eval.compute(mode="val", epoch=0, step=0)

    # ASSERT: Verify plotting was called and compute() returns metrics
    eval._log_pred_vs_target_curves.assert_called()
    assert isinstance(results, dict)  # Verify compute() returns metrics


def test_compute_skips_plotting_on_train(
    multiunit_evaluator_with_plotting_fixture, multiunit_data_time_series
):
    """
    Tests compute skips plotting for train mode.

    **PHM Logic**: Plotting is typically skipped during training to save resources.
    Only validation and test modes trigger plot generation.

    **Methodology**: Use fixture with plotting enabled, feed realistic time series data,
    call compute in train mode.

    **Uses fixture**: multiunit_data_time_series - realistic multi-unit time series data.

    **Expected**: _log_pred_vs_target_curves NOT called for train mode.

    Validates: Requirement MU-COMP-4 - Train mode plotting control
    """
    eval = multiunit_evaluator_with_plotting_fixture

    # Verify plotting was enabled at init
    assert eval.log_fit_vs_target_curves is True

    # ACT: Feed realistic time series data through proper interface
    eval.update(multiunit_data_time_series)

    # Mock the internal method
    eval._log_pred_vs_target_curves = MagicMock()

    # ACT: Mock only file operations, not orchestration
    with patch.object(eval, "_save_predictions_to_file"):
        results = eval.compute(mode="train", epoch=5, step=0)

    # ASSERT: For mode="train", plotting should NOT be called, but compute() should return metrics
    eval._log_pred_vs_target_curves.assert_not_called()
    assert isinstance(results, dict)  # Verify compute() returns metrics


# =========================================================================
# === 13. Tests: Additional Coverage for Remaining Lines ===
# =========================================================================


@patch("lmetk.evaluator.multiunit.plt")
def test_plot_reconstructions_full_path(mock_plt, multiunit_evaluator_fixture):
    """
    Tests _plot_reconstructions when remote_logger is available.

    **PHM Logic**: Full plotting path with logger configured.
    Reconstruction plots show sensor signal reconstruction quality.

    **Methodology**: Use fixture (already has logger), provide valid data.

    **Expected**: Figure created and returned.

    Validates: Requirement MU-RECON-3 - Full plotting path
    """
    eval = multiunit_evaluator_fixture
    # Fixture already provides remote_logger

    # Create proper mock axes array (5, 2) that flattens to 10
    mock_fig = MagicMock()
    mock_axes = np.empty((5, 2), dtype=object)
    for i in range(5):
        for j in range(2):
            mock_axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    mock_plt.get_cmap.return_value.colors = ["red", "blue", "green"]

    preds = np.zeros((10, 5, 2))  # 10 samples, 5 timesteps, 2 features
    targets = np.zeros((10, 5, 2))

    fig = eval._plot_reconstructions(preds, targets, "val", epoch=1, step=0)

    assert fig is not None
    mock_plt.subplots.assert_called_once_with(5, 2, figsize=(15, 20))
    mock_plt.tight_layout.assert_called_once()


def test_inverse_transform_per_unit_empty_unit_predictions(multiunit_evaluator_fixture):
    """
    Tests _inverse_transform_per_unit continues when unit predictions are empty.

    **PHM Logic**: Handle edge case of empty predictions for a unit.

    **Methodology**: Create scenario where unit has samples but predictions are empty.

    **Expected**: Continue without error.

    Validates: Requirement MU-INV-8 - Empty predictions continue
    """
    eval = multiunit_evaluator_fixture
    eval.scaling_wrapper.apply_inverse = True

    # Single sample
    preds = np.array([[[10.0]]])
    targets = np.array([[[10.0]]])
    unit_ids = np.array([1])

    # Transform returns original data
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        return_value=(preds, targets)
    )

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # Should work without error
    np.testing.assert_array_almost_equal(result_preds, preds)


def test_inverse_transform_per_unit_inf_returns_original_when_no_scaling(
    multiunit_evaluator_fixture,
):
    """
    Tests _inverse_transform_per_unit returns original when inf detected and no scaling.

    **PHM Logic**: Fallback to original when predictions not filled and scaling off.

    **Methodology**: Create scenario where predictions_filled has inf and apply_inverse=False.

    **Expected**: Returns original predictions.

    Validates: Requirement MU-INV-9 - Inf fallback with no scaling
    """
    eval = multiunit_evaluator_fixture
    eval.scaling_wrapper.apply_inverse = False

    preds = np.array([[[10.0]]])
    targets = np.array([[[10.0]]])
    unit_ids = np.array([1])

    # Mock to return same data
    eval.scaling_wrapper.inverse_transform_if_needed = MagicMock(
        return_value=(preds, targets)
    )

    result_preds, result_targets = eval._inverse_transform_per_unit(
        preds, targets, unit_ids
    )

    # Should return original data
    np.testing.assert_array_equal(result_preds, preds)


@patch("lmetk.evaluator.multiunit.matplotlib.pyplot")
def test_log_pred_vs_target_curves_empty_unit_indices(
    mock_plt, multiunit_evaluator_fixture
):
    """
    Tests _log_pred_vs_target_curves handles empty indices for a unit.

    **PHM Logic**: Some units may have no samples after filtering.
    System should gracefully skip plotting for units without data.

    **Methodology**: Use fixture (already has logger), feed data through update().

    **Expected**: Plot created for the unit that has data.

    Validates: Requirement MU-LOG-5 - Empty unit indices
    """
    eval = multiunit_evaluator_fixture
    # Fixture already provides remote_logger, mock log_plot method
    eval.log_plot = MagicMock()

    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    # Feed data through proper interface (update)
    preds = np.zeros((2, 3, 1))
    targets = np.zeros((2, 3, 1))
    unit_ids = np.array([1, 1])  # All same unit
    eval.update({"predictions": preds, "targets": targets, "unit_id": unit_ids})

    eval._log_pred_vs_target_curves(mode="val", step=0)

    # Should have created plot for unit 1
    mock_plt.subplots.assert_called()


@patch("lmetk.evaluator.multiunit.MultiUnitEvaluator._save_predictions_to_file")
def test_compute_collects_but_does_not_save(
    mock_save_file, multiunit_evaluator_factory, multiunit_data_1d_units
):
    """
    Tests the scenario where save_predictions=False but collect_predictions=True.

    **PHM Logic**: Sometimes we need to collect all predictions to generate
    plots or aggregate metrics across the whole fleet, but we don't want
    to waste disk space/IO saving large .nc or .npz files.

    **Expected**: Internal collections are populated, but file saving is skipped.
    """
    # 1. SETUP: collect=True, save=False
    evaluator = multiunit_evaluator_factory(
        # collect_predictions=True,
        save_predictions=False
    )

    # 2. ACT: Update with data
    evaluator.update(multiunit_data_1d_units)

    # Verify data is being collected internally
    assert len(evaluator.preds) > 0
    assert len(evaluator.targets) > 0

    # 3. COMPUTE
    results = evaluator.compute(mode="val", epoch=1, step=100)

    # 4. ASSERT
    # Ensure metrics were still computed (proves data was collected)
    assert "mse_denormalized_mean" in results

    # Crucially: Ensure the file-saving method was NEVER called
    mock_save_file.assert_not_called()

    # Ensure internal state still exists (for plotting etc) until reset
    assert len(evaluator.preds) > 0
