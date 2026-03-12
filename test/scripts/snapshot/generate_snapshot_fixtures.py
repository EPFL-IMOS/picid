#!/usr/bin/env python3
"""Generate synthetic data fixtures for pipeline snapshot tests.

Output: test/fixtures/snapshot/data/prognostics.npz, diagnostics.npz,
        anomaly_detection.npz, forecasting.npz

Supports:
  - test/pipeline/test_pipeline_synthetic_snapshots.py (snapshot_prognostics)
  - test/pipeline/test_pipeline_synthetic_snapshots_diagnostics.py (snapshot_diagnostics)
  - test/pipeline/test_pipeline_synthetic_snapshots_anomaly.py (snapshot_anomaly)
  - forecasting config exists but not yet in snapshot EXPERIMENTS.

Run once, commit output. Uses seed 42 for reproducibility.

Usage: uv run python test/scripts/snapshot/generate_snapshot_fixtures.py
"""
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = _PROJECT_ROOT / "test" / "fixtures" / "snapshot" / "data"
SEED = 42
N_SAMPLES = 200
N_FEATURES = 5


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Shared arrays
    features = rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)
    # Timestamps: 1-hour intervals in seconds for pd.to_datetime(unit='s') compatibility
    timestamps = (np.arange(N_SAMPLES, dtype=np.float64) * 3600).astype(np.float64)

    # Prognostics: features, rul, timestamps
    rul = np.linspace(1.0, 0.0, N_SAMPLES).reshape(-1, 1).astype(np.float32)
    np.savez(
        FIXTURES_DIR / "prognostics.npz",
        features=features,
        rul=rul,
        timestamps=timestamps,
    )

    # Diagnostics: features, target (-> fault_classification), timestamps
    target = rng.integers(0, 4, size=(N_SAMPLES, 1)).astype(np.float32)
    np.savez(
        FIXTURES_DIR / "diagnostics.npz",
        features=features,
        target=target,
        timestamps=timestamps,
    )

    # Anomaly detection: features, anomaly_detection (binary), timestamps
    anomaly = rng.integers(0, 2, size=(N_SAMPLES, 1)).astype(np.float32)
    np.savez(
        FIXTURES_DIR / "anomaly_detection.npz",
        features=features,
        anomaly_detection=anomaly,
        timestamps=timestamps,
    )

    # Forecasting: features, target, timestamps (time_features derived from timestamps in transforms)
    target_f = rng.standard_normal((N_SAMPLES, 1)).astype(np.float32)
    np.savez(
        FIXTURES_DIR / "forecasting.npz",
        features=features,
        target=target_f,
        timestamps=timestamps,
    )

    print("Fixtures written to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
