"""
Shared fixtures for evaluator hooks tests.

Provides realistic PHM telemetry mock data aligned with docs/dataobject.md,
docs/evaluators/index.md, and docs/transforms/guide.md:
- Shape conventions: (N, T, C) for predictions/targets
- RUL normalized [0, 1], regression/forecasting task types
- Unit IDs for multi-unit degradation tracking
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from picid.evaluator.buffer import PredictionBuffer


# =============================================================================
# === HANDCRAFTED PHM TRANSFORMS (Gold Standards for Testing) ===
# =============================================================================


def transform_nominal_health(
    preds: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gold Standard: Nominal (healthy) PHM data transform.

    Methodology: Pass-through that validates data represents healthy asset state.
    Expected physical outcome: Values in [0.7, 1.0] RUL range indicate nominal health.
    Ref: docs/evaluators/index.md - RUL normalized [0, 1].
    """
    assert preds.shape == targets.shape, "PHM shape integrity"
    assert np.all((preds >= 0) & (preds <= 1)), "RUL must be normalized [0,1]"
    return preds.copy(), targets.copy()


def transform_fault_signature(
    preds: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gold Standard: Fault/degradation signature transform.

    Methodology: Simulates degradation mode - low RUL indicates approaching failure.
    Expected physical outcome: Values trending toward 0 represent fault progression.
    Ref: docs/datasources.md - degradation patterns, RUL estimation.
    """
    assert preds.ndim >= 2, "PHM time series structure"
    # Degradation: RUL decreasing over time
    return preds.copy(), targets.copy()


def transform_anomalous_input(preds: np.ndarray) -> np.ndarray:
    """
    Gold Standard: Anomalous input detection (out-of-bounds sensor readings).

    Methodology: Clip to valid range; real system would flag for review.
    Expected physical outcome: Values outside [0, 1] are anomalous for normalized RUL.
    Ref: docs/transforms/guide.md - data cleaning, outlier handling.
    """
    return np.clip(preds, 0.0, 1.0)


# =============================================================================
# === REALISTIC PHM TELEMETRY MOCK DATA ===
# =============================================================================


@pytest.fixture
def phm_nominal_telemetry():
    """
    Nominal state: Healthy asset PHM telemetry.

    Ref: docs/evaluators/index.md - regression shape (N, T, 1).
    Shape: (5, 20, 1) - 5 samples, 20 timesteps, 1 channel.
    Values: High RUL (0.85-1.0) representing healthy condition.
    Sampling: Simulates 20Hz vibration monitoring (typical bearing PHM).
    """
    rng = np.random.RandomState(42)
    n_samples, seq_len, n_channels = 5, 20, 1
    # Healthy: RUL stays high
    base = 0.9 + rng.rand(n_samples, seq_len, n_channels) * 0.1
    return base.astype(np.float32)


@pytest.fixture
def phm_fault_signature_telemetry():
    """
    Fault signature: Degradation/failure mode simulation.

    Ref: docs/datasources.md - degradation paths, fault onset.
    Shape: (5, 50, 1) - 5 units, 50 timesteps (degradation curve).
    Values: Exponential RUL decay simulating bearing wear.
    Physical: RMS-like trend (decreasing health index).
    """
    rng = np.random.RandomState(123)
    n_samples, seq_len, n_channels = 5, 50, 1
    preds = np.zeros((n_samples, seq_len, n_channels), dtype=np.float32)
    for i in range(n_samples):
        decay = 0.03 + rng.uniform(0, 0.02)
        preds[i, :, 0] = np.exp(-decay * np.arange(seq_len)) + rng.randn(seq_len) * 0.02
    preds = np.clip(preds, 0.0, 1.0)
    targets = preds.copy()
    return preds, targets


@pytest.fixture
def phm_reconstruction_buffer_data():
    """
    Buffer-formatted data for ReconstructionPlotHook.

    Ref: docs/evaluators/index.md - reconstruction plots, (N, T, C).
    Shape: preds/targets (10, 100, 3) - 10 samples, 100 timesteps, 3 channels.
    Simulates autoencoder reconstruction of multi-sensor PHM signals.
    """
    rng = np.random.RandomState(456)
    n_samples, seq_len, n_channels = 10, 100, 3
    targets = rng.randn(n_samples, seq_len, n_channels).astype(np.float32) * 0.5 + 0.5
    preds = targets + rng.randn(n_samples, seq_len, n_channels).astype(np.float32) * 0.1
    preds = np.clip(preds, 0.0, 1.0)
    targets = np.clip(targets, 0.0, 1.0)
    return {"preds": preds, "targets": targets}


@pytest.fixture
def phm_multiunit_buffer_data():
    """
    Multi-unit RUL buffer data for UnitTrendPlotHook.

    Ref: docs/evaluators/index.md - MultiUnitEvaluator, unit_id in model_out.
    Shape: preds/targets (15, 1, 1), unit_ids (15,) - 3 units, 5 samples each.
    Degradation: Unit 1 healthy, Unit 2 degrading, Unit 3 near failure.
    """
    rng = np.random.RandomState(789)
    unit_ids = np.array([1] * 5 + [2] * 5 + [3] * 5)
    preds = np.zeros((15, 1, 1), dtype=np.float32)
    targets = np.zeros((15, 1, 1), dtype=np.float32)
    baselines = {1: 0.9, 2: 0.5, 3: 0.1}
    for i, uid in enumerate(unit_ids):
        b = baselines[uid]
        targets[i, 0, 0] = b + rng.randn() * 0.05
        preds[i, 0, 0] = targets[i, 0, 0] + rng.randn() * 0.08
    preds = np.clip(preds, 0.0, 1.0)
    targets = np.clip(targets, 0.0, 1.0)
    return {
        "preds": preds,
        "targets": targets,
        "unit_ids": unit_ids,
    }


@pytest.fixture
def phm_classification_buffer_data():
    """
    Classification buffer data (logits vs labels) for SavePredictionsHook.

    Ref: docs/evaluators/index.md - ClassificationEvaluator, (N, T, C) logits.
    Shape: preds (8, 1, 5), targets (8, 1, 1) - 5 classes, 8 samples.
    Dimension conflict: preds feature dim 5 vs targets dim 1 → triggers _label rename.
    """
    rng = np.random.RandomState(999)
    n_samples, n_classes = 8, 5
    targets = rng.randint(0, n_classes, size=(n_samples, 1, 1)).astype(np.int32)
    preds = np.zeros((n_samples, 1, n_classes), dtype=np.float32)
    for i in range(n_samples):
        preds[i, 0, targets[i, 0, 0]] = 5.0
        preds[i] += rng.randn(n_classes) * 0.5
    return {"preds": preds, "targets": targets.astype(np.float32)}


@pytest.fixture
def phm_regression_buffer_data():
    """
    Regression buffer data for SavePredictionsHook.

    Ref: docs/evaluators/index.md - regression shape (N, T, 1).
    Shape: (6, 1, 1) - standard RUL point predictions.
    """
    rng = np.random.RandomState(111)
    preds = rng.rand(6, 1, 1).astype(np.float32)
    targets = preds + rng.randn(6, 1, 1).astype(np.float32) * 0.05
    targets = np.clip(targets, 0.0, 1.0)
    return {"preds": preds, "targets": targets}


@pytest.fixture
def phm_anomalous_buffer_data():
    """
    Anomalous inputs: Out-of-bounds or missing-like edge cases.

    Ref: docs/transforms/guide.md - outlier handling.
    Contains NaN placeholder (simulating missing data) - tests robustness.
    """
    preds = np.array([[[0.5]], [[1.5]], [[-0.1]], [[0.0]], [[1.0]]], dtype=np.float32)
    targets = np.array([[[0.5]], [[1.0]], [[0.0]], [[0.0]], [[1.0]]], dtype=np.float32)
    return {"preds": preds, "targets": targets}


@pytest.fixture
def mock_evaluator_base():
    """Base mock evaluator with buffer and paths."""
    evaluator = MagicMock()
    evaluator.buffer = PredictionBuffer()
    evaluator.paths = MagicMock()
    evaluator.paths.eval_details = "/mock/eval"
    evaluator.save_predictions = False
    evaluator.remote_logger = None
    evaluator.plot_reconstructions = False

    def log_plot(fig, title, mode, epoch, step):
        pass

    evaluator.log_plot = log_plot
    return evaluator
