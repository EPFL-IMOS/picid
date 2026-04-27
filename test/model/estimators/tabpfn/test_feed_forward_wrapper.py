"""Canonical tests for TabPFNWrapper and helper utilities."""

import importlib
import sys
import types

import numpy as np
import pytest
import torch

_TABPFN = "tabpfn"
_WRAPPER = "picid.model.estimators.tabpfn.wrapper"
_STUBBED_MODULES = (_TABPFN, _WRAPPER)


class _StubTabPFNRegressor:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, X, y=None):
        self.fit_calls.append((X, y))
        return self

    def predict(self, X):
        self.predict_calls.append(X)
        x = X.detach().cpu().numpy() if hasattr(X, "detach") else X
        n = len(x)
        return np.full((n, 1), 2.0, dtype=np.float64)


class _StubTabPFNClassifier:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, X, y=None):
        self.fit_calls.append((X, y))
        return self

    def predict(self, X):
        self.predict_calls.append(X)
        x = X.detach().cpu().numpy() if hasattr(X, "detach") else X
        n = len(x)
        return np.zeros(n, dtype=np.int64)

    def predict_proba(self, X):
        x = X.detach().cpu().numpy() if hasattr(X, "detach") else X
        n = len(x)
        return np.tile(np.array([[0.2, 0.8]], dtype=np.float64), (n, 1))


def _install_stub_tabpfn():
    mod = types.ModuleType(_TABPFN)
    mod.TabPFNRegressor = _StubTabPFNRegressor
    mod.TabPFNClassifier = _StubTabPFNClassifier
    sys.modules[_TABPFN] = mod
    sys.modules.pop(_WRAPPER, None)
    return importlib.import_module(_WRAPPER)


def _cls():
    return _install_stub_tabpfn().TabPFNWrapper


def _generate_fn():
    return _install_stub_tabpfn().generate_and_fit_polynomial


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
    return dict(ignore_pretraining_limits=True, random_state=0, device="cpu")


def test_tabpfn_feed_forward_wrapper_reports_canonical_module():
    TabPFNWrapper = _cls()
    assert TabPFNWrapper.__module__ == "picid.model.estimators.tabpfn.wrapper"


def test_invalid_task_type_is_rejected():
    TabPFNWrapper = _cls()
    with pytest.raises(ValueError, match="not supported"):
        TabPFNWrapper(task_type="regression", **_common_kwargs())


def test_forecasting_task_uses_regressor_backbone():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="forecasting", **_common_kwargs())
    assert isinstance(w.backbone, _StubTabPFNRegressor)


def test_classification_task_uses_classifier_backbone():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="classification", **_common_kwargs())
    assert isinstance(w.backbone, _StubTabPFNClassifier)


def test_device_gpu_maps_to_cuda_string_passed_to_backbone():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="forecasting", **_common_kwargs())
    assert w.backbone.device == "cpu"
    w_gpu = TabPFNWrapper(
        task_type="forecasting",
        ignore_pretraining_limits=True,
        random_state=0,
        device="gpu",
    )
    assert w_gpu.backbone.device == "cuda"


def test_predict_regression_uses_backbone_predict_numpy():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="forecasting", **_common_kwargs())
    X = torch.randn(5, 2)
    out = w.predict(X)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (5, 1)
    np.testing.assert_allclose(out.numpy(), 2.0)


def test_predict_detaches_and_moves_tensor_before_backbone_call():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="forecasting", **_common_kwargs())
    X = torch.randn(3, 2, requires_grad=True)
    out = w.predict(X)
    assert out.shape == (3, 1)
    assert len(w.backbone.predict_calls) == 1
    received = w.backbone.predict_calls[0]
    assert isinstance(received, np.ndarray)
    np.testing.assert_allclose(received, X.detach().cpu().numpy())


def test_predict_classification_uses_predict_proba():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="classification", **_common_kwargs())
    X = torch.randn(4, 3)
    out = w.predict(X)
    assert out.shape == (4, 2)
    assert torch.allclose(out[0], torch.tensor([0.2, 0.8]))


def test_forward_squeezes_batch_calls_fit_then_predict_and_reshapes():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="forecasting", **_common_kwargs())
    batch = {
        "context": {
            "x": torch.randn(1, 3, 2),
            "y": torch.randn(1, 4, 2),
        },
        "target": {
            "x": torch.randn(1, 3, 1),
            "y": torch.randn(1, 4, 1),
        },
    }
    out = w.forward(batch)
    assert set(out.keys()) == {"predictions", "targets"}
    assert out["targets"].shape == (4, 1)
    assert out["predictions"].shape == (4, 1)
    assert len(w.backbone.fit_calls) == 1
    xf, yf = w.backbone.fit_calls[0]
    np.testing.assert_array_equal(xf, batch["context"]["x"].squeeze(0).numpy())
    np.testing.assert_array_equal(yf, batch["target"]["x"].squeeze(0).numpy())
    assert len(w.backbone.predict_calls) == 1


def test_forward_classification_branch_predict_reshaped_to_target():
    TabPFNWrapper = _cls()
    w = TabPFNWrapper(task_type="classification", **_common_kwargs())
    batch = {
        "context": {"x": torch.randn(1, 2, 3), "y": torch.randn(1, 5, 3)},
        "target": {"x": torch.randn(1, 2, 1), "y": torch.randn(1, 5, 1)},
    }
    out = w.forward(batch)
    assert out["targets"].shape == (5, 1)
    assert out["predictions"].shape == (5, 1)
    assert torch.all(out["predictions"] == 0.0)


def test_generate_and_fit_polynomial_is_deterministic_and_shapes():
    generate_and_fit_polynomial = _generate_fn()
    x_raw, y_vals, coeffs_err, y_true = generate_and_fit_polynomial(
        total_points=100,
        num_context=5,
        skip=10,
        seed=123,
        noise_std=0.01,
    )
    assert x_raw.shape == (100,)
    assert y_vals.shape == (100,)
    assert coeffs_err.shape == (5, 6)
    assert y_true.shape == (5,)
    x_raw2, y_vals2, ce2, yt2 = generate_and_fit_polynomial(
        total_points=100,
        num_context=5,
        skip=10,
        seed=123,
        noise_std=0.01,
    )
    np.testing.assert_array_equal(x_raw, x_raw2)
    np.testing.assert_array_equal(y_vals, y_vals2)
    np.testing.assert_array_equal(coeffs_err, ce2)
    np.testing.assert_array_equal(y_true, yt2)
