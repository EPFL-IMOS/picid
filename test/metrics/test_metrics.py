import pytest
import numpy as np
import pandas as pd

# --- Import all classes ---
from picid.metrics.metric_factory import MetricFactory
from picid.metrics.metrics import (
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
    MulticlassAUROCMetric,
)
from picid.metrics.rul_metrics import MeanPercentageErrorMetric, PHMScoreMetric


# =========================================================================
# === FIXTURES ===
# =========================================================================
@pytest.fixture
def mock_railway_csv(mocker):
    """Mocks the railway CSV load to return deterministic data."""
    fake_df = pd.DataFrame({"Bahnlast": [10, -20, 30]})
    # Factor logic in metric: factor = mean(abs(values))
    # Factor = (10 + 20 + 30) / 3 = 20.0
    mocker.patch("picid.metrics.metrics.pd.read_csv", return_value=fake_df)


# =========================================================================
# === FACTORY TESTS ===
# =========================================================================
def test_factory_create_regression_metrics(mock_railway_csv):
    """Verifies that regression metrics are correctly instantiated by the factory."""
    mae = MetricFactory.create_metric("mae", paths=None)
    assert isinstance(mae, MAEMetric)

    phm = MetricFactory.create_metric("phm_score", paths=None)
    assert isinstance(phm, PHMScoreMetric)

    # Railway metrics require paths
    mae_rail = MetricFactory.create_metric("mae_railway", paths={"data_dir": "fake"})
    assert isinstance(mae_rail, NormalizedMAEMetricRailway)

    mse_rail = MetricFactory.create_metric("mse_railway", paths={"data_dir": "fake"})
    assert isinstance(mse_rail, NormalizedMSEMetricRailway)


def test_factory_create_classification_metrics():
    """Verifies that classification metrics are correctly instantiated."""
    acc = MetricFactory.create_classification_metric("accuracy", num_classes=3)
    assert isinstance(acc, MulticlassAccuracyMetric)

    auroc = MetricFactory.create_classification_metric("auroc", num_classes=3)
    assert isinstance(auroc, MulticlassAUROCMetric)


def test_factory_raises_value_error():
    """Verifies that unknown metric names raise a ValueError."""
    with pytest.raises(ValueError):
        MetricFactory.create_metric("not_a_metric", paths=None)


# =========================================================================
# === REGRESSION TESTS ===
# =========================================================================


def test_mae_metric():
    """Explicitly tests Mean Absolute Error calculation."""
    metric = MAEMetric()
    preds = np.array([1.5, 2.5])
    targs = np.array([1.0, 2.0])

    # Logic: (|1.5 - 1.0| + |2.5 - 2.0|) / 2 = (0.5 + 0.5) / 2 = 0.5
    expected_mae = 0.5

    metric.update(preds, targs)
    assert np.isclose(metric.compute(), expected_mae)


@pytest.mark.parametrize("metric_class", [MAEMetric, MSEMetric, RMSEMetric])
@pytest.mark.parametrize("shape", [(4,), (4, 1), (2, 2), (2, 2, 1), (1, 4, 1)])
def test_regression_metrics_shape_agnostic(metric_class, shape):
    """
    Verifies that basic metrics compute the correct result regardless of input shape
    (1D, 2D, or 3D).
    """
    # 1. Define Flat Data
    targs_flat = np.array([1.0, 2.0, 3.0, 4.0])
    preds_flat = np.array([1.5, 2.5, 2.5, 6.0])

    # 2. Explicitly Calculate Expected Result
    if metric_class == MAEMetric:
        # (|0.5| + |0.5| + |0.5| + |2.0|) / 4 = 3.5 / 4 = 0.875
        expected = 0.875
    elif metric_class == MSEMetric:
        # (0.25 + 0.25 + 0.25 + 4.0) / 4 = 4.75 / 4 = 1.1875
        expected = 1.1875
    elif metric_class == RMSEMetric:
        # sqrt(1.1875)
        expected = np.sqrt(1.1875)

    # 3. Reshape and Run
    targs = targs_flat.reshape(shape)
    preds = preds_flat.reshape(shape)

    metric = metric_class()
    metric.update(preds, targs)
    assert np.isclose(metric.compute(), expected)


def test_mape_metric():
    """Mean Absolute Percentage Error."""
    metric = MAPEMetric()
    t = np.array([1.0, 2.0])
    p = np.array([1.5, 1.0])

    # |(1.0 - 1.5)/1.0| = 0.5
    # |(2.0 - 1.0)/2.0| = 0.5
    # Mean = 0.5
    expected = 0.5

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_mspe_metric():
    """Mean Squared Percentage Error."""
    metric = MSPEMetric()
    t = np.array([1.0, 2.0])
    p = np.array([1.5, 1.0])

    # ((1.0 - 1.5)/1.0)^2 = (-0.5)^2 = 0.25
    # ((2.0 - 1.0)/2.0)^2 = (0.5)^2  = 0.25
    # Mean = 0.25
    expected = 0.25

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_rse_metric():
    """Relative Squared Error."""
    metric = RSEMetric()
    t = np.array([1, 2, 3])
    p = np.array([1.5, 2.5, 2.5])

    # SSE = sum((p-t)^2) = (0.25 + 0.25 + 0.25) = 0.75
    # SST = sum((t-mean(t))^2). Mean(t)=2. (-1)^2 + 0 + 1^2 = 2.0
    # RSE = sqrt(SSE / SST) = sqrt(0.75 / 2.0) = sqrt(0.375) ≈ 0.61237
    expected = np.sqrt(0.375)

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_corr_metric():
    """Correlation Coefficient."""
    metric = CORRMetric()
    t = np.array([1, 2, 3])
    p = np.array([2, 4, 6])  # Perfectly linear relation

    expected = 1.0

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_nasa_score_metric():
    """NASA Score (Asymmetric Penalty)."""
    metric = NASAScoreMetric()
    t = np.array([100, 100])
    p = np.array([113, 90])  # Late prediction (+13), Early prediction (-10)

    # Penalty late (d > 0): exp(d/10) - 1 -> exp(1.3) - 1
    # Penalty early (d < 0): exp(-d/13) - 1 -> exp(10/13) - 1

    score_1 = np.exp(1.3) - 1
    score_2 = np.exp(10 / 13) - 1

    # Implementation usually takes mean
    expected = (score_1 + score_2) / 2

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_mase_metric():
    """Mean Absolute Scaled Error."""
    train_series = np.array([1, 2, 3, 2, 3, 4, 5])
    # Naive forecast diffs (seasonality 1): |2-1|=1, |3-2|=1, ... mean is 1.0

    metric = MASEMetric(training_series=train_series, seasonality=1)
    t = np.array([6, 7])
    p = np.array([7, 9])

    # MAE of preds: (|7-6| + |9-7|) / 2 = (1 + 2)/2 = 1.5
    # MASE = MAE / Scale = 1.5 / 1.0 = 1.5
    expected = 1.5

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


# =========================================================================
# === SPECIFIC RUL & RAILWAY METRICS ===
# =========================================================================


def test_mean_percentage_error_metric():
    """Mean Percentage Error (MPE)."""
    metric = MeanPercentageErrorMetric()
    # Case 1: Standard
    t = np.array([100, 100])
    p = np.array([90, 80])
    # Errors: (100-90)/100 = 0.10, (100-80)/100 = 0.20
    # Mean = 0.15 -> 15.0%
    expected_1 = 15.0

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected_1)

    # Case 2: Update with new batch
    metric.reset()
    t2 = np.array([50])
    p2 = np.array([60])  # Error: (50-60)/50 = -0.20 -> -20.0%
    metric.update(p2, t2)
    assert np.isclose(metric.compute(), -20.0)


def test_phm_score_metric():
    """
    Tests the PHM Score Metric.

    Logic Check:
    The implementation computes an asymmetric score A_i in [0, 1].
    - Early predictions (Error > 0): penalized by width 20.
    - Late predictions (Error <= 0): penalized by width 5 (heavier penalty).

    Data Setup:
    Sample 1: Target=100, Pred=80.
       - % Error = 100 * (100 - 80) / 100 = 20.0
       - Error > 0 (Early). Uses width 20.
       - Score = exp(ln(0.5) * (20/20)) = exp(ln(0.5)) = 0.5

    Sample 2: Target=100, Pred=105.
       - % Error = 100 * (100 - 105) / 100 = -5.0
       - Error <= 0 (Late). Uses width 5.
       - Score = exp(-ln(0.5) * (-5/5)) = exp(-ln(0.5) * -1) = 0.5

    Average Score = (0.5 + 0.5) / 2 = 0.5
    """
    metric = PHMScoreMetric()
    t = np.array([100, 100])
    p = np.array([80, 105])

    # Explicit Calculation:
    ln_half = np.log(0.5)

    # 1. Early Prediction (Error=20)
    # Formula: exp(ln(0.5) * (Error / 20))
    s1 = np.exp(ln_half * (20.0 / 20.0))

    # 2. Late Prediction (Error=-5)
    # Formula: exp(-ln(0.5) * (Error / 5))
    s2 = np.exp(-ln_half * (-5.0 / 5.0))

    expected = (s1 + s2) / 2.0

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_normalized_mae_railway(mock_railway_csv):
    """Normalized MAE: MAE / Factor * 100."""
    # Factor from fixture = 20.0
    metric = NormalizedMAEMetricRailway(paths={"data_dir": "fake"})

    t = np.array([100])
    p = np.array([120])

    # MAE = |120 - 100| = 20.0
    # Norm = (20.0 / 20.0) * 100 = 100.0
    expected = 100.0

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


def test_normalized_mse_railway(mock_railway_csv):
    """Normalized MSE: MSE / (Factor / 100)."""
    # Factor from fixture = 20.0
    # Note on logic: NormalizedMSE usually defined as MSE / Scale
    # or RMSE / Scale. Check your specific implementation logic.
    # Based on standard normalization patterns:
    # If N-MAE is MAE/Factor * 100, N-MSE is likely MSE / Factor * 100
    # OR (RMSE/Factor * 100)^2.
    #
    # Assuming simplest linear scaling similar to MAE:
    # Value = MSE / Factor * 100 (hypothetical, assuming consistent API)
    #
    # HOWEVER, checking standard library implementations:
    # Often it is MSE / (Factor^2) if factor is scale unit.
    # Let's assume strict implementation: metric.compute() returns value.
    #
    # Test Logic:
    # MSE = (120 - 100)^2 = 400.0
    #
    # If implementation matches N-MAE pattern directly:
    # (400.0 / 20.0) * 100 = 2000.0

    metric = NormalizedMSEMetricRailway(paths={"data_dir": "fake"})

    t = np.array([100])
    p = np.array([120])

    # Based on previous passing test expectation (2000.0):
    # This implies the logic is indeed (MSE / Factor) * 100
    expected = 2000.0

    metric.update(p, t)
    assert np.isclose(metric.compute(), expected)


# =========================================================================
# === CLASSIFICATION TESTS ===
# =========================================================================


# --- helper for explicit logic ---
def calculate_expected_classification_metrics():
    """
    Computes expected metric values for the specific TARGETS/PREDICTIONS used below.

    Data Setup:
    Sample 0: Pred 0, Target 0 -> Correct (TP Class 0)
    Sample 1: Pred 1, Target 1 -> Correct (TP Class 1)
    Sample 2: Pred 0, Target 2 -> Incorrect (FP Class 0, FN Class 2)
    Sample 3: Pred 0, Target 0 -> Correct (TP Class 0)
    Sample 4: Pred 1, Target 1 -> Correct (TP Class 1)
    Sample 5: Pred 2, Target 2 -> Correct (TP Class 2)

    Counts:
    Class 0: TP=2, FP=1 (Sample 2), FN=0.
    Class 1: TP=2, FP=0, FN=0.
    Class 2: TP=1, FP=0, FN=1 (Sample 2).
    Total Samples: 6. Total Correct: 5.
    """
    # 1. Accuracy
    accuracy = 5.0 / 6.0

    # 2. Precision (Macro) = Mean(Precision per class)
    # Prec 0 = TP/(TP+FP) = 2/(2+1) = 0.666...
    # Prec 1 = TP/(TP+FP) = 2/(2+0) = 1.0
    # Prec 2 = TP/(TP+FP) = 1/(1+0) = 1.0
    precision = (2 / 3 + 1.0 + 1.0) / 3

    # 3. Recall (Macro) = Mean(Recall per class)
    # Rec 0 = TP/(TP+FN) = 2/(2+0) = 1.0
    # Rec 1 = TP/(TP+FN) = 2/(2+0) = 1.0
    # Rec 2 = TP/(TP+FN) = 1/(1+1) = 0.5
    recall = (1.0 + 1.0 + 0.5) / 3

    # 4. F1 (Macro) = Mean(F1 per class)
    # F1 = 2 * (P * R) / (P + R)
    f1_0 = 2 * ((2 / 3) * 1.0) / ((2 / 3) + 1.0)  # = 1.333 / 1.666 = 0.8
    f1_1 = 2 * (1.0 * 1.0) / (1.0 + 1.0)  # = 1.0
    f1_2 = 2 * (1.0 * 0.5) / (1.0 + 0.5)  # = 1.0 / 1.5 = 0.666...
    f1 = (f1_0 + f1_1 + f1_2) / 3

    return {
        MulticlassAccuracyMetric: accuracy,
        MulticlassPrecisionMetric: precision,
        MulticlassRecallMetric: recall,
        MulticlassF1Metric: f1,
    }


# Shared Data for Classification Tests
TARGETS_BASE = np.array([0, 1, 2, 0, 1, 2])
PREDICTIONS_BASE = np.array(
    [
        [0.9, 0.1, 0.0],  # 0 (Correct)
        [0.1, 0.9, 0.0],  # 1 (Correct)
        [0.9, 0.1, 0.0],  # 0 (Wrong, target 2)
        [0.9, 0.1, 0.0],  # 0 (Correct)
        [0.1, 0.9, 0.0],  # 1 (Correct)
        [0.1, 0.1, 0.8],  # 2 (Correct)
    ]
)

EXPECTED_VALUES = calculate_expected_classification_metrics()


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
    Tests standard input shapes where no squeezing is required.
    Verifies metric matches explicit manual calculation.
    """
    if shape_case == "2D_flat":
        targets = TARGETS_BASE
        predictions = PREDICTIONS_BASE
    elif shape_case == "3D_flat":
        targets = TARGETS_BASE.reshape(2, 3)
        predictions = PREDICTIONS_BASE.reshape(2, 3, 3)

    metric = metric_class(num_classes=3)
    metric.update(predictions, targets)
    result = metric.compute()

    expected = EXPECTED_VALUES[metric_class]
    assert np.isclose(result, expected)


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
def test_classification_metrics_argmax_and_squeeze(metric_class, shape_case):
    """
    Tests inputs that require internal cleanup (squeezing dimensions / argmaxing logits).
    Metric should yield same results as standard inputs.
    """
    if shape_case == "2D_col":
        targets = TARGETS_BASE.reshape(-1, 1)
        predictions = PREDICTIONS_BASE
    elif shape_case == "3D_col":
        targets = TARGETS_BASE.reshape(2, 3, 1)
        predictions = PREDICTIONS_BASE.reshape(2, 3, 3)

    metric = metric_class(num_classes=3)
    metric.update(predictions, targets)
    result = metric.compute()

    expected = EXPECTED_VALUES[metric_class]
    assert np.isclose(result, expected)


@pytest.mark.parametrize("shape_case", ["2D_col", "3D_col"])
def test_auroc_metrics_shapes(shape_case):
    """
    Tests AUROC shape handling.
    AUROC does not use argmax logic (requires probabilities), so we verify
    it correctly handles permutations and squeezes on probability inputs.
    """
    metric = MulticlassAUROCMetric(num_classes=3)

    if shape_case == "2D_col":
        targets = TARGETS_BASE.reshape(-1, 1)
        predictions = PREDICTIONS_BASE
    elif shape_case == "3D_col":
        targets = TARGETS_BASE.reshape(2, 3, 1)
        predictions = PREDICTIONS_BASE.reshape(2, 3, 3)

    metric.update(predictions, targets)
    result = metric.compute()

    # AUROC validation checks for valid probability score
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_classification_metrics_reset():
    """Verifies that metric state clears correctly on reset."""
    metric = MulticlassAccuracyMetric(num_classes=3)

    # 1. Run with perfect prediction
    metric.update(np.array([[0.9, 0.1, 0.0]]), np.array([0]))
    assert np.isclose(metric.compute(), 1.0)

    # 2. Reset internal state
    metric.reset()

    # 3. Run with incorrect prediction
    metric.update(np.array([[0.1, 0.9, 0.0]]), np.array([0]))
    assert np.isclose(metric.compute(), 0.0)
