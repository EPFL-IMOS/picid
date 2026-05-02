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


# ---------------------------------------------------------------------------
# Tests for the real AutoGluonWrapper (picid.model.autogluon.autogluon_wrapper)
# ---------------------------------------------------------------------------

_AG_REAL_MODULE = "picid.model.autogluon.autogluon_wrapper"
_AG_DEPS = [
    "autogluon",
    "autogluon.tabular",
    "autogluon.features",
    "autogluon.features.generators",
    "picid.model.autogluon.multi_label_predictor",
]


class _FakeTabularPredictor:
    def __init__(self, label, problem_type, path):
        self.label = label
        self.problem_type = problem_type
        self.is_fit = False

    def fit(self, df, **kwargs):
        self.is_fit = True


class _FakeMultilabelPredictor:
    def __init__(self, labels, problem_type, path):
        self.labels = labels
        self.problem_type = problem_type
        self.is_fit = False

    def fit(self, df, **kwargs):
        self.is_fit = True


class _FakeGenerator:
    def __init__(self):
        pass


class _FakeBulkGenerator:
    def __init__(self, generators):
        self.generators = generators


@pytest.fixture
def _real_ag_module():
    """Import the real AutoGluonWrapper with stubbed autogluon dependencies."""
    ag = types.ModuleType("autogluon")
    ag_tab = types.ModuleType("autogluon.tabular")
    ag_tab.TabularPredictor = _FakeTabularPredictor
    ag_feat = types.ModuleType("autogluon.features")
    ag_feat_gen = types.ModuleType("autogluon.features.generators")
    ag_feat_gen.AsTypeFeatureGenerator = _FakeGenerator
    ag_feat_gen.IdentityFeatureGenerator = _FakeGenerator
    ag_feat_gen.BulkFeatureGenerator = _FakeBulkGenerator
    ag.tabular = ag_tab
    ag.features = ag_feat

    mlp_mod = types.ModuleType("picid.model.autogluon.multi_label_predictor")
    mlp_mod.MultilabelTabularPredictor = _FakeMultilabelPredictor

    saved = {name: sys.modules.get(name) for name in _AG_DEPS + [_AG_REAL_MODULE]}
    sys.modules.update(
        {
            "autogluon": ag,
            "autogluon.tabular": ag_tab,
            "autogluon.features": ag_feat,
            "autogluon.features.generators": ag_feat_gen,
            "picid.model.autogluon.multi_label_predictor": mlp_mod,
        }
    )
    sys.modules.pop(_AG_REAL_MODULE, None)
    mod = importlib.import_module(_AG_REAL_MODULE)
    mod.AutoGluonWrapper.__abstractmethods__ = frozenset()

    yield mod

    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _make_batch_loader(n_targets=1):
    batch = {
        "context": {"features_seq_x": torch.zeros(2, 10, 5)},
        "target": {"target_seq_x": torch.zeros(2, 10, n_targets)},
    }
    return iter([batch])


@pytest.mark.unit
class TestAutoGluonWrapperInit:
    """Tests for AutoGluonWrapper.__init__ and _initialize_backbone."""

    def test_single_target_creates_tabular_predictor(self, _real_ag_module):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            model_dict=OmegaConf.create({}),
        )
        assert isinstance(w.backbone, _FakeTabularPredictor)
        assert w.n_targets == 1

    def test_multi_target_creates_multilabel_predictor(self, _real_ag_module):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=3),
            model_dict=OmegaConf.create({}),
        )
        assert isinstance(w.backbone, _FakeMultilabelPredictor)
        assert w.n_targets == 3

    def test_classification_task_sets_classification_problem_type(
        self, _real_ag_module
    ):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            task_type="classification",
            model_dict=OmegaConf.create({}),
        )
        assert w.backbone.problem_type == "classification"

    def test_regression_task_sets_binary_problem_type(self, _real_ag_module):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            task_type="regression",
            model_dict=OmegaConf.create({}),
        )
        assert w.backbone.problem_type == "binary"

    def test_default_generators_two_items(self, _real_ag_module):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            model_dict=OmegaConf.create({}),
        )
        assert isinstance(w.feature_generators, _FakeBulkGenerator)
        assert len(w.feature_generators.generators) == 2

    def test_custom_generator_list_passed_through(self, _real_ag_module):
        from omegaconf import OmegaConf

        class _MyGen:
            def __init__(self):
                pass

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            model_dict=OmegaConf.create({}),
            generator_list=[_MyGen()],
        )
        assert len(w.feature_generators.generators) == 1


@pytest.mark.unit
class TestAutoGluonWrapperFit:
    """Tests for AutoGluonWrapper.fit."""

    def test_fit_calls_backbone_fit(self, _real_ag_module):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            model_dict=OmegaConf.create({}),
        )
        w.fit(np.zeros((5, 3)), np.zeros((5, 1)))
        assert w.backbone.is_fit is True

    def test_fit_reinitializes_backbone_when_already_fit(self, _real_ag_module):
        from omegaconf import OmegaConf

        w = _real_ag_module.AutoGluonWrapper(
            train_dataloader=_make_batch_loader(n_targets=1),
            model_dict=OmegaConf.create({}),
        )
        w.backbone.is_fit = True
        old_backbone = w.backbone
        w.fit(np.zeros((5, 3)), np.zeros((5, 1)))
        assert w.backbone is not old_backbone
