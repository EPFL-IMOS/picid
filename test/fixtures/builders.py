"""Deterministic array and dict builders: ``seed`` in, payload out."""

from __future__ import annotations

import numpy as np

from test.fixtures.rng import numpy_rs


def make_regression_batch(*, seed: int, batch_size: int = 5) -> dict[str, np.ndarray]:
    """Basic normalized RUL-style regression batch (predictions + noisy targets)."""
    rs = numpy_rs(seed)
    preds = np.linspace(0.9, 0.1, batch_size).reshape(batch_size, 1, 1)
    targets = preds + rs.randn(batch_size, 1, 1) * 0.05
    targets = np.clip(targets, 0.0, 1.0)
    return {
        "predictions": preds.astype(np.float32),
        "targets": targets.astype(np.float32),
    }


def make_synthetic_features_float32(
    *, seed: int = 42, time_steps: int = 100, n_features: int = 8
) -> np.ndarray:
    """Dense float32 feature matrix ``(T, F)`` in ``[0, 1)`` (first draw from ``seed``)."""
    return numpy_rs(seed).rand(time_steps, n_features).astype(np.float32)


def make_forecasting_target_continuing_features(
    features: np.ndarray, *, seed: int = 42
) -> np.ndarray:
    """Forecasting target column matching the RNG stream after ``make_synthetic_features_float32``."""
    t, f = features.shape
    rs = numpy_rs(seed)
    rs.rand(t, f)
    return rs.randn(t, 1).astype(np.float32)


def make_fault_classification_targets_continuing_features(
    features: np.ndarray, *, seed: int = 42, n_classes: int = 5
) -> np.ndarray:
    """Fault labels matching the stream after feature draws (same as forecasting path + randint)."""
    t, f = features.shape
    rs = numpy_rs(seed)
    rs.rand(t, f)
    return rs.randint(0, n_classes, size=(t, 1), dtype=np.int32).astype(np.float32)


def make_synthetic_fit_predict_3d(
    *,
    seed: int = 42,
    n_tasks: int = 4,
    n_samples: int = 50,
    n_features: int = 6,
    n_targets: int = 1,
) -> dict[str, np.ndarray]:
    rs = numpy_rs(seed)
    x = rs.randn(n_tasks, n_samples, n_features).astype(np.float32)
    y = rs.randn(n_tasks, n_samples, n_targets).astype(np.float32)
    return {"X": x, "y": y}


def make_synthetic_image_like_uint8(
    *, seed: int = 42, size: tuple[int, int, int, int] = (10, 32, 32, 3)
) -> np.ndarray:
    return numpy_rs(seed).randint(0, 256, size=size, dtype=np.uint8)


def make_synthetic_normalized_float32(
    *, seed: int = 42, rows: int = 20, cols: int = 5
) -> np.ndarray:
    return numpy_rs(seed).rand(rows, cols).astype(np.float32)


def make_standard_normal_2d(*, seed: int, n_rows: int, n_cols: int) -> np.ndarray:
    return numpy_rs(seed).randn(n_rows, n_cols)


def split_container_unit_seed(base_seed: int, split: str, unit: int, field: int) -> int:
    """Stable derived seed for split-container units (features vs target)."""
    split_id = {"train": 1, "val": 2, "test": 3}[split]
    return base_seed + split_id * 10_000 + unit * 100 + field
