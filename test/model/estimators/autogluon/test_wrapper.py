"""Canonical tests for FitPredictAutoGluonWrapper."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_AG_MODULE = "picid.model.autogluon.autogluon_wrapper"
_WRAPPER_MODULE = "picid.model.estimators.autogluon.wrapper"
_STUBBED_MODULES = (_AG_MODULE, _WRAPPER_MODULE)


class _StubAutoGluonWrapper:
    """Minimal sklearn-like backbone; classification if ag_problem is set."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.problem = kwargs.get("ag_problem", "regression")
        self._fitted = False

    def fit(self, x, y):
        self._fitted = True

    def predict(self, x):
        n = np.asarray(x).shape[0]
        if self.problem == "classification":
            p = np.tile(np.array([[0.2, 0.8]], dtype=np.float64), (n, 1))
            return p
        return np.arange(n, dtype=np.float64).reshape(n, 1)


def _install_stub_autogluon_wrapper():
    fake = types.ModuleType(_AG_MODULE)
    fake.AutoGluonWrapper = _StubAutoGluonWrapper
    sys.modules[_AG_MODULE] = fake
    sys.modules.pop(_WRAPPER_MODULE, None)
    return importlib.import_module(_WRAPPER_MODULE)


def _fresh_wrapper():
    return _install_stub_autogluon_wrapper().FitPredictAutoGluonWrapper


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_autogluon_wrapper_reports_canonical_module():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    assert (
        FitPredictAutoGluonWrapper.__module__
        == "picid.model.estimators.autogluon.wrapper"
    )


def test_invalid_task_type_is_rejected():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    with pytest.raises(ValueError, match="not supported"):
        FitPredictAutoGluonWrapper(task_type="forecasting")


def test_allows_multi_target_is_true():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(task_type="regression")
    assert w.allows_multi_target is True


def test_regression_fit_predict_uses_stub_backbone():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(task_type="regression")
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    w.fit(X, y)
    pred = w.predict(X)
    assert pred.shape == (5, 1)
    assert pred.dtype in (torch.float32, torch.float64)


def test_classification_predict_returns_two_columns_when_stub_backbone_is_classification():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(
        task_type="classification",
        ag_problem="classification",
    )
    X = torch.randn(4, 3)
    y = torch.zeros(4, 1, dtype=torch.long)
    w.fit(X, y)
    out = w.predict(X)
    assert out.shape == (4, 2)
    assert np.allclose(out.sum(dim=1).numpy(), 1.0, atol=1e-5)


def test_regression_backbone_selection_default_stub_problem():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(task_type="regression")
    assert w.backbone.problem == "regression"


def test_classification_task_with_regression_stub_still_runs_regression_shaped_predict():
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(task_type="classification")
    assert w.backbone.problem == "regression"
    X = torch.randn(3, 2)
    y = torch.zeros(3, 1)
    w.fit(X, y)
    pred = w.predict(X)
    assert pred.shape == (3, 1)


def test_serialize_load_restores_fitted_backbone(tmp_path):
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(
        task_type="regression",
        model_cache_path=str(tmp_path),
    )
    X = torch.ones(3, 2)
    y = torch.zeros(3, 1)
    w.fit(X, y)
    assert w.backbone._fitted is True
    path = w.serialize_model("ag-task")
    assert Path(path).exists()
    w2 = FitPredictAutoGluonWrapper(
        task_type="regression",
        model_cache_path=str(tmp_path),
    )
    w2.load_model("ag-task")
    assert w2.backbone._fitted is True
    assert torch.equal(w.predict(X), w2.predict(X))


def test_relative_model_cache_path_is_not_duplicated(tmp_path, monkeypatch):
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    monkeypatch.chdir(tmp_path)
    relative_cache = Path("rel-cache")
    w = FitPredictAutoGluonWrapper(
        task_type="regression",
        model_cache_path=str(relative_cache),
    )
    X = torch.ones(3, 2)
    y = torch.zeros(3, 1)
    w.fit(X, y)
    path = w.serialize_model("ag-task")
    expected = relative_cache / "ag-task.joblib"
    assert Path(path) == expected
    assert expected.exists()
    assert not (relative_cache / "rel-cache" / "ag-task.joblib").exists()
    loaded_path = w.load_model("ag-task")
    assert Path(loaded_path) == expected


def test_load_model_missing_file_raises(tmp_path):
    FitPredictAutoGluonWrapper = _fresh_wrapper()
    w = FitPredictAutoGluonWrapper(
        task_type="regression",
        model_cache_path=str(tmp_path),
    )
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        w.load_model("missing-ag")
