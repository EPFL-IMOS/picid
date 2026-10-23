"""Canonical tests for FitPredictTabDPTWrapper."""

import importlib
import sys
import types
from pathlib import Path

import cloudpickle
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
        self.prediction_calls = []

    def fit(self, X, y):
        self._fitted = True
        return self

    def predict(self, X, n_ensembles=8, context_size=2048, seed=None):
        self.prediction_calls.append((seed, n_ensembles))
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
    assert FitPredictTabDPTWrapper.__module__ == "picid.model.estimators.tabdpt.wrapper"


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


@pytest.mark.parametrize("seed", [0, 72, 88, None])
def test_regression_forwards_prediction_seed_with_eight_members(seed):
    wrapper = _cls()(device="cpu", task_type="regression", random_state=seed)
    X = torch.ones(3, 2)
    wrapper.fit(X, torch.zeros(3, 1))

    wrapper.predict(X)

    assert wrapper.backbone.prediction_calls == [(seed, 8)]


@pytest.mark.parametrize("batched", [False, True])
def test_repeated_and_batched_predictions_keep_the_same_seed(batched):
    wrapper = _cls()(
        device="cpu",
        task_type="regression",
        random_state=72,
        yield_strategy=batched,
        yield_batch_size=2,
    )
    X = torch.ones(5, 2)
    wrapper.fit(X, torch.zeros(5, 1))

    wrapper.predict(X)
    wrapper.predict(X)

    expected_calls = 6 if batched else 2
    assert wrapper.backbone.prediction_calls == [(72, 8)] * expected_calls


def test_configured_seed_does_not_change_classification_call():
    wrapper = _cls()(device="cpu", task_type="classification", random_state=72)
    X = torch.ones(3, 2)
    wrapper.fit(X, torch.zeros(3, 1))

    # The classifier double accepts no seed or ensemble arguments.
    result = wrapper.predict(X)

    torch.testing.assert_close(result, torch.tensor([[0.4, 0.6]] * 3))


@pytest.mark.parametrize("saved_seed", [0, 72, None])
def test_checkpoint_restores_prediction_seed_over_constructor_seed(
    tmp_path, saved_seed
):
    Wrapper = _cls()
    original = Wrapper(
        device="cpu",
        task_type="regression",
        random_state=saved_seed,
        model_cache_path=str(tmp_path),
    )
    X = torch.ones(3, 2)
    original.fit(X, torch.zeros(3, 1))
    original.serialize_model("seeded")
    restored = Wrapper(
        device="cpu",
        task_type="regression",
        random_state=101,
        model_cache_path=str(tmp_path),
    )

    restored.load_model("seeded")
    restored.predict(X)

    assert restored.backbone.prediction_calls == [(saved_seed, 8)]
    restored.serialize_model("resaved")
    again = Wrapper(
        device="cpu", task_type="regression", model_cache_path=str(tmp_path)
    )
    again.load_model("resaved")
    again.predict(X)
    assert again.backbone.prediction_calls[-1] == (saved_seed, 8)


@pytest.mark.parametrize("configured_seed", [0, 101, None])
def test_legacy_checkpoint_uses_configured_seed_after_loading_seeded_model(
    tmp_path, configured_seed
):
    Wrapper = _cls()
    original = Wrapper(
        device="cpu",
        task_type="regression",
        random_state=72,
        model_cache_path=str(tmp_path),
    )
    X = torch.ones(3, 2)
    original.fit(X, torch.zeros(3, 1))
    original.serialize_model("seeded")
    with (tmp_path / "legacy.cloudpickle").open("wb") as handle:
        cloudpickle.dump(original.backbone, handle)
    restored = Wrapper(
        device="cpu",
        task_type="regression",
        random_state=configured_seed,
        model_cache_path=str(tmp_path),
    )
    restored.load_model("seeded")
    restored.load_model("legacy")

    restored.predict(X)

    assert restored.backbone.prediction_calls == [(configured_seed, 8)]


def test_refitting_loaded_model_restores_constructor_seed(tmp_path):
    Wrapper = _cls()
    original = Wrapper(
        device="cpu",
        task_type="regression",
        random_state=72,
        model_cache_path=str(tmp_path),
    )
    X = torch.ones(3, 2)
    original.fit(X, torch.zeros(3, 1))
    original.serialize_model("seeded")
    restored = Wrapper(
        device="cpu",
        task_type="regression",
        random_state=101,
        model_cache_path=str(tmp_path),
    )
    restored.load_model("seeded")

    restored.fit(X, torch.zeros(3, 1))
    restored.predict(X)

    assert restored.backbone.prediction_calls == [(101, 8)]


def test_classification_checkpoint_remains_a_bare_estimator(tmp_path):
    wrapper = _cls()(
        device="cpu",
        task_type="classification",
        random_state=72,
        model_cache_path=str(tmp_path),
    )
    X = torch.ones(3, 2)
    wrapper.fit(X, torch.zeros(3, 1))
    path = wrapper.serialize_model("classifier")

    with open(path, "rb") as handle:
        saved = cloudpickle.load(handle)

    assert isinstance(saved, _StubTabDPTClassifier)


def test_checkpoint_rejects_unknown_format_version(tmp_path):
    wrapper = _cls()(
        device="cpu", task_type="regression", model_cache_path=str(tmp_path)
    )
    with (tmp_path / "future.cloudpickle").open("wb") as handle:
        cloudpickle.dump(
            {
                "format": "picid.tabdpt",
                "version": 999,
                "backbone": _StubTabDPTRegressor(),
                "prediction_seed": 72,
            },
            handle,
        )

    with pytest.raises(ValueError, match="Unsupported TabDPT checkpoint version"):
        wrapper.load_model("future")
