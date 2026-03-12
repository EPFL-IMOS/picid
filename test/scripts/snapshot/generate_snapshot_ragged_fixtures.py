#!/usr/bin/env python3
"""Generate synthetic ragged data fixtures for pipeline snapshot tests.

Output: test/fixtures/snapshot/data/ragged_prognostics.pkl

Supports:
  - test/data/datasources/test_synthetic_ragged_loader.py (SyntheticRaggedFromFileLoader)
  - test/pipeline/test_pipeline_synthetic_snapshots.py (prognostics_ragged experiment)

Run once, commit output. Uses seed 42 for reproducibility.

Usage: uv run python test/scripts/snapshot/generate_snapshot_ragged_fixtures.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = _PROJECT_ROOT / "test" / "fixtures" / "snapshot" / "data"
SEED = 42
N_UNITS = 3


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Per unit: variable-length sequences (n_cycles varies)
    lengths = [50 + int(rng.integers(0, 30)) for _ in range(N_UNITS)]
    features_list = []
    rul_list = []
    unit_id_list = []
    for i, L in enumerate(lengths):
        feat = rng.standard_normal((L, 3)).astype(np.float32)
        features_list.append(feat)
        rul_list.append(np.linspace(1.0, 0.0, L).astype(np.float32))
        unit_id_list.append(np.full((L, 1), i, dtype=np.int64))

    data = {
        "features": features_list,
        "rul": rul_list,
        "target": rul_list,
        "unit_id": unit_id_list,
    }
    out_path = FIXTURES_DIR / "ragged_prognostics.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    print("Ragged fixtures written to", out_path)


if __name__ == "__main__":
    main()
