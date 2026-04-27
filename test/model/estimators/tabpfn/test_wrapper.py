"""Tests for FitPredictTabPFNWrapper with a stub ``tabpfn`` package (no TabPFN install)."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_TABPFN = "tabpfn"
_WRAPPER = "picid.model.estimators.tabpfn.wrapper"
_STUBBED_MODULES = (_TABPFN, _WRAPPER)


class _StubTabPFNRegressor:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")

    def fit(self, X, y=None):
        return self

    def predict(self, X, output_type="mean"):
        x = X.numpy() if hasattr(X, "numpy") else X
        n = len(x)
        if output_type == "full":
            return {
                "mean": torch.ones(n, 1, dtype=torch.float32) * 3.0,
                "criterion": torch.tensor(0.25),
                "logits": torch.zeros(n, 2),
            }
        return np.ones((n, 1), dtype=np.float64) * 2.0


class _StubTabPFNClassifier:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        n = len(X)
        return np.tile(np.array([[0.3, 0.7]]), (n, 1))


def _install_stub_tabpfn():
    mod = types.ModuleType(_TABPFN)
    mod.TabPFNRegressor = _StubTabPFNRegressor
    mod.TabPFNClassifier = _StubTabPFNClassifier
    sys.modules[_TABPFN] = mod
    sys.modules.pop(_WRAPPER, None)
    return importlib.import_module(_WRAPPER)


def _cls():
    return _install_stub_tabpfn().FitPredictTabPFNWrapper


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _common_kwargs():
    return dict(ignore_pretraining_limits=True, random_state=0)


def test_tabpfn_wrapper_reports_canonical_module():
    FitPredictTabPFNWrapper = _cls()
    assert (
        FitPredictTabPFNWrapper.__module__ == "picid.model.estimators.tabpfn.wrapper"
    )


def test_invalid_task_type_is_rejected():
    FitPredictTabPFNWrapper = _cls()
    with pytest.raises(ValueError, match="not supported"):
        FitPredictTabPFNWrapper(
            device="cpu",
            task_type="forecasting",
            **_common_kwargs(),
        )


def test_allows_multi_target_is_false():
    FitPredictTabPFNWrapper = _cls()
    w = FitPredictTabPFNWrapper(
        device="cpu", task_type="regression", **_common_kwargs()
    )
    assert w.allows_multi_target is False


def test_regression_predict_without_full_outputs_path():
    FitPredictTabPFNWrapper = _cls()
    w = FitPredictTabPFNWrapper(
        device="cpu", task_type="regression", **_common_kwargs()
    )
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    w.fit(X, y)
    pred = w.predict(X)
    assert pred.shape == (5, 1)
    assert pred.dtype == torch.float32


def test_classification_uses_predict_proba():
    FitPredictTabPFNWrapper = _cls()
    w = FitPredictTabPFNWrapper(
        device="cpu", task_type="classification", **_common_kwargs()
    )
    X = torch.randn(4, 3)
    y = torch.zeros(4, 1, dtype=torch.long)
    w.fit(X, y)
    out = w.predict(X)
    assert out.shape == (4, 2)


def test_full_outputs_path_writes_predictions_pt_and_returns_mean(tmp_path):
    FitPredictTabPFNWrapper = _cls()
    out_dir = tmp_path / "full_out"
    out_dir.mkdir()
    w = FitPredictTabPFNWrapper(
        device="cpu",
        task_type="regression",
        full_outputs_path=str(out_dir),
        output_type="full",
        **_common_kwargs(),
    )
    X = torch.randn(3, 2)
    y = torch.randn(3, 1)
    w.fit(X, y)
    pred = w.predict(X)
    saved = out_dir / "predictions.pt"
    assert saved.exists()
    bundle = torch.load(saved, map_location="cpu")
    assert "mean" in bundle
    assert pred.shape == (3, 1)
    assert torch.allclose(pred, torch.ones(3, 1) * 3.0)


def test_serialize_and_load_roundtrip(tmp_path):
    FitPredictTabPFNWrapper = _cls()
    w = FitPredictTabPFNWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(tmp_path),
        **_common_kwargs(),
    )
    X = torch.randn(4, 2)
    y = torch.randn(4, 1)
    w.fit(X, y)
    path = w.serialize_model("tpfn")
    assert Path(path).exists()
    w2 = FitPredictTabPFNWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(tmp_path),
        **_common_kwargs(),
    )
    w2.load_model("tpfn")
    assert type(w2.backbone) is type(w.backbone)


def test_relative_model_cache_path_is_not_duplicated(tmp_path, monkeypatch):
    FitPredictTabPFNWrapper = _cls()
    monkeypatch.chdir(tmp_path)
    relative_cache = Path("rel-cache")
    w = FitPredictTabPFNWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(relative_cache),
        **_common_kwargs(),
    )
    X = torch.randn(4, 2)
    y = torch.randn(4, 1)
    w.fit(X, y)
    path = w.serialize_model("tpfn")
    expected = relative_cache / "tpfn.tabpfn_fit"
    assert Path(path) == expected
    assert expected.exists()
    assert not (relative_cache / "rel-cache" / "tpfn.tabpfn_fit").exists()
    loaded_path = w.load_model("tpfn")
    assert Path(loaded_path) == expected


def test_load_model_missing_file_raises(tmp_path):
    FitPredictTabPFNWrapper = _cls()
    w = FitPredictTabPFNWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(tmp_path),
        **_common_kwargs(),
    )
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        w.load_model("nope")
