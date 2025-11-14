import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock

# --- Import all classes to be tested ---
# (Adjust these import paths to match your library structure)
from lmetk.metrics.base import AbstractMetric, AbstractFileSystemMetric
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.metrics.metrics import (
    MAEMetric,
    MSEMetric,
    RMSEMetric,
    MAPEMetric,
    MSPEMetric,
    RSEMetric,
    CORRMetric,
    NASAScoreMetric,
    MASEMetric,
    NormalizedMAEMetricRailway,
    NormalizedMSEMetricRailway,
    MulticlassAccuracyMetric,
    MulticlassPrecisionMetric,
    MulticlassRecallMetric,
    MulticlassF1Metric,
)
from lmetk.metrics.rul_metrics import MeanPercentageErrorMetric, PHMScoreMetric

# =========================================================================
# === 1. Fixtures for Mocking ===
# =========================================================================


@pytest.fixture
def mock_railway_csv(mocker):
    """
    Mocks the pd.read_csv call inside the Normalized...Railway metrics.
    """
    # Create the fake DataFrame it will return
    fake_df = pd.DataFrame({"Bahnlast": [10, -20, 30]})
    # The factor will be: mean(abs([10, -20, 30])) = mean([10, 20, 30]) = 20

    # Patch pd.read_csv *as it is seen by* the metrics.py module
    mocker.patch("lmetk.metrics.metrics.pd.read_csv", return_value=fake_df)


# =========================================================================
# === 2. Test MetricFactory ===
# =========================================================================
# (These tests passed and are unchanged)


def test_factory_create_regression_metrics(mock_railway_csv):
    """Tests that the factory can create all known regression metrics."""
    # Test a simple one
    mae = MetricFactory.create_metric("mae", paths=None)
    assert isinstance(mae, MAEMetric)

    # Test a RUL-specific one
    phm = MetricFactory.create_metric("phm_score", paths=None)
    assert isinstance(phm, PHMScoreMetric)

    # Test a filesystem-dependent one
    mae_rail = MetricFactory.create_metric("mae_railway", paths={"data_dir": "fake"})
    assert isinstance(mae_rail, NormalizedMAEMetricRailway)

    # Test case-insensitivity
    rmse_upper = MetricFactory.create_metric("RMSE", paths=None)
    assert isinstance(rmse_upper, RMSEMetric)


def test_factory_create_classification_metrics():
    """Tests that the factory can create all known classification metrics."""
    acc = MetricFactory.create_classification_metric("accuracy", num_classes=3)
    assert isinstance(acc, MulticlassAccuracyMetric)

    f1_upper = MetricFactory.create_classification_metric("F1", num_classes=3)
    assert isinstance(f1_upper, MulticlassF1Metric)


def test_factory_raises_value_error():
    """Tests that the factory raises an error for unknown metrics."""
    with pytest.raises(ValueError, match="Unknown metric 'not_a_metric'"):
        MetricFactory.create_metric("not_a_metric", paths=None)

    with pytest.raises(
        ValueError, match="Unknown classification metric 'not_a_metric'"
    ):
        MetricFactory.create_classification_metric("not_a_metric", num_classes=3)


# =========================================================================
# === 3. Test Concrete Regression Metrics ===
# =========================================================================
# (These tests passed and are unchanged)

# --- Test data for simple regression metrics ---
TARGS_1 = np.array([1.0, 2.0, 3.0])
PREDS_1 = np.array([1.5, 2.5, 2.5])
TARGS_2 = np.array([4.0])
PREDS_2 = np.array([6.0])


def test_mae_metric():
    """Tests MAE with batching and reset logic."""
    metric = MAEMetric()
    metric.update(PREDS_1, TARGS_1)
    assert np.isclose(metric.compute(), 0.5)
    metric.reset()
    assert metric.total_count == 0
    with pytest.raises(ValueError):
        metric.compute()
    metric.update(PREDS_1, TARGS_1)
    metric.update(PREDS_2, TARGS_2)
    assert np.isclose(metric.compute(), 0.875)


@pytest.mark.parametrize(
    "metric_class, expected_result",
    [
        (MAEMetric, 0.875),
        (MSEMetric, 1.1875),
        (RMSEMetric, 1.0897247),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (4,),  # 1D: (N,)
        (4, 1),  # 2D: (N, D) where D=1
        (2, 2),  # 2D: (N, D)
        (2, 2, 1),  # 3D: (N, T, D)
        (1, 4, 1),  # 3D: (N, T, D)
    ],
)
def test_regression_metrics_shape_agnostic(metric_class, expected_result, shape):
    """
    Tests that basic accumulation metrics (MAE, MSE, RMSE) are
    agnostic to input shape (1D, 2D, or 3D).
    """
    targs_flat = np.array([1.0, 2.0, 3.0, 4.0])
    preds_flat = np.array([1.5, 2.5, 2.5, 6.0])

    targs = targs_flat.reshape(shape)
    preds = preds_flat.reshape(shape)

    metric = metric_class()

    metric.update(preds, targs)
    result = metric.compute()

    assert np.isclose(result, expected_result)


def test_mape_metric():
    metric = MAPEMetric()
    t = np.array([1.0, 2.0])
    p = np.array([1.5, 1.0])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 0.5)
    metric.reset()
    t_zero = np.array([0.0, 1.0])
    p_zero = np.array([1.0, 3.0])
    metric.update(p_zero, t_zero)
    assert np.isclose(metric.compute(), 2.0)


def test_mspe_metric():
    metric = MSPEMetric()
    t = np.array([1.0, 2.0])
    p = np.array([1.5, 1.0])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 0.25)


def test_rse_metric():
    metric = RSEMetric()
    t = np.array([1, 2, 3])
    p = np.array([1.5, 2.5, 2.5])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 0.6123724)


def test_corr_metric():
    metric = CORRMetric()
    t1 = np.array([1, 2, 3])
    p1 = np.array([2, 4, 6])
    metric.update(p1, t1)
    assert np.isclose(metric.compute(), 1.0)
    metric.reset()
    p2 = np.array([6, 4, 2])
    metric.update(p2, t1)
    assert np.isclose(metric.compute(), -1.0)
    metric.reset()
    p3 = np.array([1, 1, 1])
    metric.update(p3, t1)
    assert np.isclose(metric.compute(), 0.0)
    metric.reset()
    metric.update(np.array([2, 4]), np.array([1, 2]))
    metric.update(np.array([6]), np.array([3]))
    assert np.isclose(metric.compute(), 1.0)


def test_nasa_score_metric():
    metric = NASAScoreMetric()
    t = np.array([100, 100])
    p = np.array([113, 90])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 1.9137011007838451)


def test_mase_metric():
    train_series = np.array([1, 2, 3, 2, 3, 4, 5])
    seasonality = 1
    metric = MASEMetric(training_series=train_series, seasonality=seasonality)
    t = np.array([6, 7])
    p = np.array([7, 9])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 1.5)


# =========================================================================
# === 4. Test RUL-Specific Metrics ===
# =========================================================================
# (These tests passed and are unchanged)


def test_mean_percentage_error_metric():
    metric = MeanPercentageErrorMetric()
    t = np.array([100, 100])
    p = np.array([90, 80])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 15.0)
    metric.update(np.array([110]), np.array([100]))
    assert np.isclose(metric.compute(), (0.2 / 3) * 100)


def test_phm_score_metric():
    metric = PHMScoreMetric()
    t = np.array([100, 100])
    p = np.array([80, 105])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 0.5)


# =========================================================================
# === 5. Test FileSystem (Railway) Metrics ===
# =========================================================================
# (These tests passed and are unchanged)


def test_normalized_mae_railway(mock_railway_csv):
    metric = NormalizedMAEMetricRailway(paths={"data_dir": "fake"})
    assert metric.factor == 20
    t = np.array([100])
    p = np.array([120])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 100.0)


def test_normalized_mse_railway(mock_railway_csv):
    metric = NormalizedMSEMetricRailway(paths={"data_dir": "fake"})
    assert metric.factor == 20
    t = np.array([100])
    p = np.array([120])
    metric.update(p, t)
    assert np.isclose(metric.compute(), 2000.0)


# =========================================================================
# === 6. Test Classification Metrics (UPDATED) ===
# =========================================================================

# --- Base data for classification tests ---
# (6 samples total)
TARGETS_BASE = np.array([0, 1, 2, 0, 1, 2])
PREDICTIONS_BASE = np.array(
    [
        [0.9, 0.1, 0.0],  # -> 0. Correct
        [0.1, 0.9, 0.0],  # -> 1. Correct
        [0.9, 0.1, 0.0],  # -> 0. WRONG (target=2)
        [0.9, 0.1, 0.0],  # -> 0. Correct
        [0.1, 0.9, 0.0],  # -> 1. Correct
        [0.1, 0.1, 0.8],  # -> 2. Correct
    ]
)
# Expected result for micro-averaging: 5 Correct / 6 Total = 0.8333...
EXPECTED_RESULT = 5.0 / 6.0


@pytest.mark.parametrize(
    "metric_class",
    [
        MulticlassAccuracyMetric,
        MulticlassPrecisionMetric,
        MulticlassRecallMetric,
        MulticlassF1Metric,
    ],
)
@pytest.mark.parametrize("shape_case", ["2D_flat", "3D_flat"])
def test_classification_metrics_good_shapes(metric_class, shape_case):
    """
    Tests that classification metrics work with valid shape combinations.
    - (N, C) vs (N,)
    - (N, T, C) vs (N, T)
    """
    # --- Arrange ---
    if shape_case == "2D_flat":
        # preds (N, C), targets (N,)
        targets = TARGETS_BASE
        predictions = PREDICTIONS_BASE
    elif shape_case == "3D_flat":
        # N=2, T=3
        # preds (N, T, C), targets (N, T)
        targets = TARGETS_BASE.reshape(2, 3)
        predictions = PREDICTIONS_BASE.reshape(2, 3, 3)

    metric = metric_class(num_classes=3)

    # --- Act ---
    metric.update(predictions, targets)
    result = metric.compute()

    # --- Assert ---
    assert np.isclose(result, EXPECTED_RESULT)


@pytest.mark.parametrize(
    "metric_class",
    [
        MulticlassAccuracyMetric,
        MulticlassPrecisionMetric,
        MulticlassRecallMetric,
        MulticlassF1Metric,
    ],
)
@pytest.mark.parametrize("shape_case", ["2D_col", "3D_col"])
def test_classification_metrics_bad_shapes(metric_class, shape_case):
    """
    Tests that classification metrics FAIL correctly with invalid shape combinations.
    - (N, C) vs (N, 1)
    - (N, T, C) vs (N, T, 1)
    """
    # --- Arrange ---
    if shape_case == "2D_col":
        # preds (N, C), targets (N, 1)
        targets = TARGETS_BASE.reshape(-1, 1)
        predictions = PREDICTIONS_BASE
    elif shape_case == "3D_col":
        # N=2, T=3
        # preds (N, T, C), targets (N, T, 1)
        targets = TARGETS_BASE.reshape(2, 3, 1)
        predictions = PREDICTIONS_BASE.reshape(2, 3, 3)

    metric = metric_class(num_classes=3)

    # --- Act & Assert ---
    # We assert that a ValueError is raised because the shapes are incompatible.
    with pytest.raises(
        ValueError, match="Either `preds` and `target` both should have"
    ):
        metric.update(predictions, targets)


def test_classification_metrics_reset():
    """Tests the reset logic for a classification metric."""
    metric = MulticlassAccuracyMetric(num_classes=3)

    # Run with correct data
    metric.update(np.array([[0.9, 0.1, 0.0]]), np.array([0]))  # -> 0  # -> 0
    assert np.isclose(metric.compute(), 1.0)

    # Reset
    metric.reset()

    # Run with incorrect data
    metric.update(np.array([[0.1, 0.9, 0.0]]), np.array([0]))  # -> 1  # -> 0
    assert np.isclose(metric.compute(), 0.0)
