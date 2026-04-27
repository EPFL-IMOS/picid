"""Canonical tests for the Isolation Forest estimator wrapper."""

from pathlib import Path

import numpy as np
import pytest
import torch

from picid.model.estimators.isolation_forest.wrapper import (
    FitPredictIsolationForestWrapper,
)


def _features(n: int = 8, f: int = 3, *, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, f)


def test_isolation_forest_wrapper_reports_canonical_module():
    assert (
        FitPredictIsolationForestWrapper.__module__
        == "picid.model.estimators.isolation_forest.wrapper"
    )


def test_invalid_task_type_is_rejected():
    with pytest.raises(ValueError, match="not supported"):
        FitPredictIsolationForestWrapper(task_type="forecasting")


def test_allows_multi_target_is_false():
    w = FitPredictIsolationForestWrapper(task_type="anomaly_detection")
    assert w.allows_multi_target is False


def test_fit_ignores_labels_and_predict_returns_two_column_logits():
    X = _features(12, 4, seed=1)
    y = torch.randint(0, 10, (12, 1))
    w = FitPredictIsolationForestWrapper(
        task_type="anomaly_detection", n_estimators=20, random_state=0
    )
    w.fit(X, y)
    out = w.predict(X)
    assert out.shape == (12, 2)
    assert out.dtype == torch.float32


def test_serialize_and_load_roundtrip(tmp_path):
    X = _features(10, 2, seed=2)
    y = torch.zeros(10, 1)
    w = FitPredictIsolationForestWrapper(
        task_type="anomaly_detection",
        n_estimators=15,
        random_state=1,
        model_cache_path=str(tmp_path),
    )
    w.fit(X, y)
    path = w.serialize_model("iso-task")
    assert Path(path).exists()
    loaded = w.load_model("iso-task")
    raw = loaded.decision_function(X.numpy())
    expected = w.backbone.decision_function(X.numpy())
    assert np.allclose(raw, expected)
    assert Path(path).parent == tmp_path


def test_relative_model_cache_path_is_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative_cache = Path("rel-cache")
    X = _features(10, 2, seed=3)
    y = torch.zeros(10, 1)
    w = FitPredictIsolationForestWrapper(
        task_type="anomaly_detection",
        n_estimators=10,
        random_state=2,
        model_cache_path=str(relative_cache),
    )
    w.fit(X, y)
    path = w.serialize_model("iso-task")
    expected = relative_cache / "iso-task.joblib"
    assert Path(path) == expected
    assert expected.exists()
    assert not (relative_cache / "rel-cache" / "iso-task.joblib").exists()
    loaded = w.load_model("iso-task")
    assert hasattr(loaded, "decision_function")


def test_load_model_missing_file_raises(tmp_path):
    w = FitPredictIsolationForestWrapper(
        task_type="classification",
        model_cache_path=str(tmp_path),
        random_state=0,
    )
    with pytest.raises((FileNotFoundError, OSError)):
        w.load_model("definitely-missing-task")


def test_serialize_model_task_id_none_raises():
    w = FitPredictIsolationForestWrapper(task_type="anomaly_detection", n_estimators=5)
    with pytest.raises(ValueError, match="no task_id"):
        w.serialize_model(None)


def test_load_model_task_id_none_raises():
    w = FitPredictIsolationForestWrapper(task_type="anomaly_detection", n_estimators=5)
    with pytest.raises(ValueError, match="no task_id"):
        w.load_model(None)


def test_call_fit_reinit_on_fit_invokes_reinit_backbone(monkeypatch):
    X = _features(8, 3, seed=4)
    y = torch.zeros(8, 1)
    w = FitPredictIsolationForestWrapper(
        task_type="anomaly_detection", n_estimators=8, random_state=3
    )
    w.reinit_on_fit = True
    calls = []

    monkeypatch.setattr(w, "_reinit_backbone", lambda: calls.append(1))
    w.fit(X, y)
    assert calls == [1]
