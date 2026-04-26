#!/usr/bin/env python3
"""Tutorial: canonical isolation-forest fit-predict basics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from picid.model.estimators.isolation_forest.wrapper import (
    FitPredictIsolationForestWrapper,
)


def main():
    # Create X (100, 5) and y (100,) - y is dummy for IsolationForest (unsupervised)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.zeros(100, dtype=np.float32)  # ignored by IsolationForest

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = FitPredictIsolationForestWrapper(task_type="anomaly_detection")
    model.fit(X_t, y_t)

    pred = model.predict(X_t)

    # pred is 2D tensor; IsolationForest returns (N, 2) logits [normal, anomaly]
    assert isinstance(pred, torch.Tensor), "pred must be torch.Tensor"
    assert pred.ndim == 2, "pred must be 2D"
    assert pred.shape[0] == 100, f"Expected 100 rows, got {pred.shape[0]}"
    assert pred.shape[1] in (1, 2), f"Expected 1 or 2 cols, got {pred.shape[1]}"

    print("OK")


if __name__ == "__main__":
    main()
