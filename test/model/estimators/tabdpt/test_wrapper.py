"""Canonical tests for FitPredictTabDPTWrapper."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_TABDPT = "tabdpt"
_WRAPPER = "picid.model.estimators.tabdpt.wrapper"
_STUBBED_MODULES = (_TABDPT, _WRAPPER)


class _StubTabDPTRegressor:
    def __init__(self, **kwargs):
        self.inf_batch_size = kwargs.get("inf_batch_size", 128)
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True
        return self

    def predict(self, X):
        n = len(X)
        return np.linspace(0.0, 1.0, n).reshape(n, 1)


class _StubTabDPTClassifier:
    def __init__(self, **kwargs):
        self.inf_batch_size = kwargs.get("inf_batch_size", 128)
        self._fitted = False

    def fit(self, X, y):
        self._fitted = True
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1, keepdims=True)

    def predict_proba(self, X):
        n = len(X)
        return np.tile(np.array([[0.4, 0.6]], dtype=np.float64), (n, 1))


def _install_stub_tabdpt():
    mod = types.ModuleType(_TABDPT)
    mod.TabDPTRegressor = _StubTabDPTRegressor
    mod.TabDPTClassifier = _StubTabDPTClassifier
    sys.modules[_TABDPT] = mod
    sys.modules.pop(_WRAPPER, None)
    return importlib.import_module(_WRAPPER)


def _cls():
    return _install_stub_tabdpt().FitPredictTabDPTWrapper


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_tabdpt_wrapper_reports_canonical_module():
    FitPredictTabDPTWrapper = _cls()
    assert (
        FitPredictTabDPTWrapper.__module__
        == "picid.model.estimators.tabdpt.wrapper"
    )


def test_invalid_task_type_is_rejected():
    FitPredictTabDPTWrapper = _cls()
    with pytest.raises(ValueError, match="not supported"):
        FitPredictTabDPTWrapper(device="cpu", task_type="forecasting")


def test_allows_multi_target_is_false():
    FitPredictTabDPTWrapper = _cls()
    w = FitPredictTabDPTWrapper(device="cpu", task_type="regression")
    assert w.allows_multi_target is False


def test_regression_partial_factory_instantiates_regressor_on_fit():
    FitPredictTabDPTWrapper = _cls()
    w = FitPredictTabDPTWrapper(
        device="cpu", task_type="regression", yield_batch_size=16
    )
    assert w.backbone_factory is not None
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    w.fit(X, y)
    assert type(w.backbone).__name__ == "_StubTabDPTRegressor"
    pred = w.predict(X)
    assert pred.shape == (5, 1)


def test_classification_partial_factory_and_predict_proba():
    FitPredictTabDPTWrapper = _cls()
    w = FitPredictTabDPTWrapper(
        device="cpu", task_type="classification", yield_batch_size=8
    )
    assert w.backbone_factory.func is _StubTabDPTClassifier
    X = torch.randn(4, 3)
    y = torch.zeros(4, 1, dtype=torch.long)
    w.fit(X, y)
    assert type(w.backbone).__name__ == "_StubTabDPTClassifier"
    out = w.predict(X)
    assert out.shape == (4, 2)


def test_serialize_and_load_cloudpickle_roundtrip(tmp_path):
    FitPredictTabDPTWrapper = _cls()
    w = FitPredictTabDPTWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(tmp_path),
    )
    X = torch.randn(3, 2)
    y = torch.randn(3, 1)
    w.fit(X, y)
    path = w.serialize_model("tabdpt")
    assert Path(path).exists()
    w2 = FitPredictTabDPTWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(tmp_path),
    )
    w2.load_model("tabdpt")
    assert torch.allclose(w.predict(X), w2.predict(X))


def test_relative_model_cache_path_is_not_duplicated(tmp_path, monkeypatch):
    FitPredictTabDPTWrapper = _cls()
    monkeypatch.chdir(tmp_path)
    relative_cache = Path("rel-cache")
    w = FitPredictTabDPTWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(relative_cache),
    )
    X = torch.randn(3, 2)
    y = torch.randn(3, 1)
    w.fit(X, y)
    path = w.serialize_model("tabdpt")
    expected = relative_cache / "tabdpt.cloudpickle"
    assert Path(path) == expected
    assert expected.exists()
    assert not (relative_cache / "rel-cache" / "tabdpt.cloudpickle").exists()
    loaded_path = w.load_model("tabdpt")
    assert Path(loaded_path) == expected


def test_load_model_missing_file_raises(tmp_path):
    FitPredictTabDPTWrapper = _cls()
    w = FitPredictTabDPTWrapper(
        device="cpu",
        task_type="regression",
        model_cache_path=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        w.load_model("ghost")
