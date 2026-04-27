"""Canonical tests for FitPredictCarteWrapper."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

_CARTE = "carte_ai"
_CARTE_CFG = "carte_ai.configs"
_CARTE_DIR = "carte_ai.configs.directory"
_HF = "huggingface_hub"
_WRAPPER = "picid.model.estimators.carte.wrapper"
_STUBBED_MODULES = (_CARTE, _CARTE_CFG, _CARTE_DIR, _HF, _WRAPPER)

_download_log: list[tuple[str, str]] = []


class _StubCARTERegressor:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.kwargs = kwargs
        self._fitted = False

    def fit(self, X=None, y=None, **kw):
        self._fitted = True
        return self

    def predict(self, X):
        n = len(X)
        return np.full((n, 1), 42.0, dtype=np.float64)


class _StubTable2GraphTransformer:
    def __init__(self, fasttext_model_path=None):
        self.fasttext_model_path = fasttext_model_path
        self.events: list[str] = []

    def fit_transform(self, X, y=None):
        self.events.append("fit_transform")
        return X

    def transform(self, X):
        self.events.append("transform")
        return X


def _install_carte_stubs(fake_pretrained: str = "/stub/pretrained.pt"):
    def hf_hub_download(repo_id: str, filename: str, **kwargs):
        _download_log.append((repo_id, filename))
        return "/stub/fasttext-fake.bin"

    hf = types.ModuleType(_HF)
    hf.hf_hub_download = hf_hub_download
    sys.modules[_HF] = hf

    carte_pkg = types.ModuleType(_CARTE)
    carte_pkg.__path__ = []

    cfg_pkg = types.ModuleType(_CARTE_CFG)
    cfg_pkg.__path__ = []
    sys.modules[_CARTE_CFG] = cfg_pkg

    dir_mod = types.ModuleType(_CARTE_DIR)
    dir_mod.config_directory = {"pretrained_model": fake_pretrained}
    sys.modules[_CARTE_DIR] = dir_mod

    carte_pkg.CARTERegressor = _StubCARTERegressor
    carte_pkg.Table2GraphTransformer = _StubTable2GraphTransformer
    sys.modules[_CARTE] = carte_pkg

    sys.modules.pop(_WRAPPER, None)
    return importlib.import_module(_WRAPPER)


def _cls():
    _download_log.clear()
    return _install_carte_stubs().FitPredictCarteWrapper


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    yield
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _make_wrapper(task_type: str = "regression", **extra):
    FitPredictCarteWrapper = _cls()
    return FitPredictCarteWrapper(
        num_model=1,
        disable_pbar=True,
        random_state=0,
        device="cpu",
        n_jobs=1,
        task_type=task_type,
        **extra,
    )


def test_carte_wrapper_reports_canonical_module():
    FitPredictCarteWrapper = _cls()
    assert FitPredictCarteWrapper.__module__ == "picid.model.estimators.carte.wrapper"


def test_hf_hub_download_stub_used_for_fasttext():
    _make_wrapper()
    assert _download_log == [("hi-paris/fastText", "cc.en.300.bin")]


def test_invalid_task_type_is_rejected():
    FitPredictCarteWrapper = _cls()
    with pytest.raises(ValueError, match="not supported"):
        FitPredictCarteWrapper(
            num_model=1,
            disable_pbar=True,
            random_state=0,
            device="cpu",
            n_jobs=1,
            task_type="forecasting",
        )


def test_allows_multi_target_is_false():
    w = _make_wrapper()
    assert w.allows_multi_target is False


def test_preprocessing_bridge_fit_transform_then_transform():
    w = _make_wrapper()
    assert w.preprocessor.fasttext_model_path == "/stub/fasttext-fake.bin"
    X = torch.randn(6, 2)
    y = torch.randn(6, 1)
    w.fit(X, y)
    assert w.preprocessor.events == ["fit_transform"]
    w.predict(X)
    assert w.preprocessor.events == ["fit_transform", "transform"]


def test_regression_predict_shape():
    w = _make_wrapper(task_type="regression")
    X = torch.randn(4, 3)
    y = torch.randn(4, 1)
    w.fit(X, y)
    pred = w.predict(X)
    assert pred.shape == (4, 1)


def test_classification_task_supported_same_backbone():
    w = _make_wrapper(task_type="classification")
    X = torch.randn(3, 2)
    y = torch.zeros(3, 1)
    w.fit(X, y)
    pred = w.predict(X)
    assert pred.shape == (3, 1)


def test_serialize_and_load_joblib(tmp_path):
    w = _make_wrapper(model_cache_path=str(tmp_path))
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    w.fit(X, y)
    path = w.serialize_model("carte")
    assert Path(path).exists()
    w2 = _make_wrapper(model_cache_path=str(tmp_path))
    w2.load_model("carte")
    assert np.allclose(w.predict(X).numpy(), w2.predict(X).numpy())


def test_relative_model_cache_path_is_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative_cache = Path("rel-cache")
    w = _make_wrapper(model_cache_path=str(relative_cache))
    X = torch.randn(5, 2)
    y = torch.randn(5, 1)
    w.fit(X, y)
    path = w.serialize_model("carte")
    expected = relative_cache / "carte.tabpfn_fit"
    assert Path(path) == expected
    assert expected.exists()
    assert not (relative_cache / "rel-cache" / "carte.tabpfn_fit").exists()
    loaded_path = w.load_model("carte")
    assert Path(loaded_path) == expected


def test_load_model_missing_file_raises(tmp_path):
    w = _make_wrapper(model_cache_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        w.load_model("missing")
