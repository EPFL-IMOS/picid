"""Unit tests for shared behavior in picid.model.adapters.base."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from picid.model.adapters.base import (
    AbstractFeedForwardTrainingWrapper,
    AbstractFeedForwardWrapper,
    AbstractFitPredictWrapper,
)


class _DummyFitPredictWrapper(AbstractFitPredictWrapper):
    """Minimal concrete wrapper for exercising AbstractFitPredictWrapper."""

    @property
    def allows_multi_target(self) -> bool:
        return False

    def serialize_model(self, task_id):
        return None

    def load_model(self, task_id):
        return None


class _NumpyBackbone:
    def __init__(self, y_predict=None):
        self._y_predict = y_predict
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1

    def predict(self, X):
        if self._y_predict is not None:
            return self._y_predict
        n = X.shape[0]
        return np.arange(n, dtype=np.float64).reshape(n, 1)


class _TorchBackbone:
    def fit(self, X, y):
        pass

    def predict(self, X):
        n = X.shape[0]
        return torch.ones(n, 3, dtype=torch.float32)


class _TorchFirstFeatureBackbone:
    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[:, 0:1].clone()


class _FirstFeatureBackbone:
    """Predict first feature column as (n, 1) numpy — stable under batched predict."""

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[:, 0:1].detach().numpy().astype(np.float64)


class _NoPredictBackbone:
    def fit(self, X, y):
        pass


class _CallableWithSklearnAPI:
    """Callable object that also exposes fit/predict (hits the non-factory branch rules)."""

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros((X.shape[0], 1), dtype=np.float64)

    def __call__(self):
        return 0


class _FFWrapper(AbstractFeedForwardWrapper):
    def forward(self, batch):
        return {"tag": "ff", "batch_keys": tuple(batch.keys())}


class _FFTrainingWrapper(AbstractFeedForwardTrainingWrapper):
    def forward(self, batch):
        return {"tag": "fft", "x": batch.get("x")}


class _BackboneWithOutChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 7


class _ReprFeedForwardWrapper(AbstractFeedForwardWrapper):
    """Concrete wrapper so ``AbstractFeedForwardWrapper.__repr__`` can run."""

    def __init__(self, backbone, **kwargs):
        self.dimensions = kwargs.get("dimensions", 2)
        self.residual_connections = kwargs.get("residual_connections", True)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        return {}


class _AbstractPassThrough(AbstractFitPredictWrapper):
    """Exercise abstract ``pass`` bodies in ``serialize_model`` / ``load_model`` via ``super()``."""

    @property
    def allows_multi_target(self) -> bool:
        return False

    def serialize_model(self, task_id):
        return super().serialize_model(task_id)

    def load_model(self, task_id):
        return super().load_model(task_id)


def test_adapter_base_is_the_real_implementation_module():
    assert (
        AbstractFeedForwardTrainingWrapper.__module__ == "picid.model.adapters.base"
    )


def test_invalid_backbone_missing_predict_raises_valueerror():
    with pytest.raises(ValueError, match="must implement fit and predict"):
        _DummyFitPredictWrapper(backbone=_NoPredictBackbone())


def test_callable_backbone_requires_reinit_on_fit():
    def factory():
        return _NumpyBackbone()

    with pytest.raises(AssertionError, match="reinit_on_fit must be True"):
        _DummyFitPredictWrapper(backbone=factory, reinit_on_fit=False)


def test_callable_instance_with_fit_predict_not_allowed():
    with pytest.raises(AssertionError, match="can not reinitialize on fit"):
        _DummyFitPredictWrapper(backbone=_CallableWithSklearnAPI(), reinit_on_fit=True)


def test_reinit_backbone_without_factory_raises_runtimeerror():
    w = _DummyFitPredictWrapper(backbone=_NumpyBackbone())
    with pytest.raises(RuntimeError, match="No backbone factory"):
        w._reinit_backbone()


def test_reinit_backbone_factory_returns_invalid_raises_valueerror():
    def factory():
        return _NoPredictBackbone()

    w = _DummyFitPredictWrapper(backbone=factory, reinit_on_fit=True)
    X = torch.zeros(2, 1)
    y = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="Re-initialized backbone must implement"):
        w.fit(X, y)


def test_callable_factory_reinit_on_fit_creates_fresh_backbone_each_fit():
    builds = []

    def factory():
        b = _NumpyBackbone()
        builds.append(b)
        return b

    w = _DummyFitPredictWrapper(backbone=factory, reinit_on_fit=True)
    X = torch.ones(3, 2)
    y = torch.zeros(3, 1)
    w.fit(X, y)
    first = w.backbone
    w.fit(X, y)
    second = w.backbone
    assert first is not second
    assert len(builds) == 2


def test_iterate_batches_respects_yield_batch_size():
    w = _DummyFitPredictWrapper(
        backbone=_NumpyBackbone(), yield_batch_size=2, yield_strategy=True
    )
    X = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    batches = list(w._iterate_batches(X))
    assert len(batches) == 3
    assert batches[0].shape == (2, 4)
    assert batches[1].shape == (2, 4)
    assert batches[2].shape == (1, 4)
    assert torch.equal(batches[0], X[:2].cpu())


def test_predict_converts_numpy_to_tensor_and_keeps_2d():
    w = _DummyFitPredictWrapper(backbone=_NumpyBackbone())
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = w.predict(X)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 1)
    assert out.dtype == torch.float64


def test_predict_reshapes_1d_output_to_column():
    y1d = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    w = _DummyFitPredictWrapper(backbone=_NumpyBackbone(y_predict=y1d))
    X = torch.zeros(3, 1)
    out = w.predict(X)
    assert out.shape == (3, 1)


def test_predict_invalid_output_type_raises_typeerror():
    class _BadBackbone:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return ["not", "tensor"]

    w = _DummyFitPredictWrapper(backbone=_BadBackbone())
    with pytest.raises(TypeError, match="torch.Tensor or numpy.ndarray"):
        w.predict(torch.zeros(2, 1))


def test_yield_strategy_batched_predict_concatenates_numpy_batches():
    backbone = _FirstFeatureBackbone()
    X = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expected = _DummyFitPredictWrapper(backbone=backbone).predict(X)

    w = _DummyFitPredictWrapper(
        backbone=backbone, yield_strategy=True, yield_batch_size=2
    )
    out = w.predict(X)
    assert out.shape == (4, 1)
    assert torch.allclose(out, expected)


def test_yield_strategy_batched_predict_accepts_torch_tensor_batches():
    backbone = _TorchFirstFeatureBackbone()
    X = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expected = _DummyFitPredictWrapper(backbone=backbone).predict(X)

    w = _DummyFitPredictWrapper(
        backbone=backbone, yield_strategy=True, yield_batch_size=2
    )
    out = w.predict(X)
    assert out.shape == (4, 1)
    assert torch.equal(out, expected)


def test_predict_accepts_torch_tensor_from_backbone():
    w = _DummyFitPredictWrapper(backbone=_TorchBackbone())
    X = torch.zeros(2, 5)
    out = w.predict(X)
    assert out.shape == (2, 3)
    assert (out == 1).all()


def test_predict_rejects_non_2d_outputs():
    class _ThreeDimBackbone:
        def fit(self, X, y):
            pass

        def predict(self, X):
            return torch.ones(X.shape[0], 2, 2)

    w = _DummyFitPredictWrapper(backbone=_ThreeDimBackbone())
    with pytest.raises(AssertionError, match="2-dimensional"):
        w.predict(torch.zeros(2, 3))


def test_abstract_feed_forward_wrapper_call_delegates_to_forward():
    backbone = torch.nn.Linear(2, 2)
    w = _FFWrapper(backbone=backbone, out_channels=2, num_cell_dimensions=1)
    batch = {"a": torch.zeros(1)}
    out = w(batch)
    assert out == {"tag": "ff", "batch_keys": ("a",)}


def test_abstract_feed_forward_training_wrapper_call_delegates_to_forward():
    backbone = torch.nn.Linear(2, 2)
    w = _FFTrainingWrapper(backbone=backbone, out_channels=2, num_cell_dimensions=1)
    x = torch.ones(1, 2)
    out = w({"x": x})
    assert out["tag"] == "fft"
    assert out["x"] is x


def test_both_profile_cfgs_raises_assertion():
    fit_cfg = OmegaConf.create({"wait": 0})
    pred_cfg = OmegaConf.create({"wait": 0})
    with pytest.raises(AssertionError, match="Only one of profile_fit"):
        _DummyFitPredictWrapper(
            backbone=_NumpyBackbone(),
            profile_fit_cfg=fit_cfg,
            profile_predict_cfg=pred_cfg,
        )


def test_fit_profile_path_invokes_profiler(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "wait": 0,
            "warmup": 0,
            "active": 2,
            "repeat": 1,
            "activities": ["CPU"],
            "trace_path": str(tmp_path / "fit_trace.json"),
        }
    )
    w = _DummyFitPredictWrapper(backbone=_NumpyBackbone(), profile_fit_cfg=cfg)
    monkeypatch.setattr(w, "_check_stop_profiler", lambda _c: None)
    w.fit(torch.zeros(3, 2), torch.zeros(3, 1))
    assert w.profile_started_flag is True
    assert hasattr(w, "profiler")
    w.profiler.stop()


def test_predict_yield_verbose_and_profile_predict_cfg(monkeypatch, tmp_path):
    cfg = OmegaConf.create(
        {
            "wait": 0,
            "warmup": 0,
            "active": 2,
            "repeat": 1,
            "activities": ["CPU"],
            "trace_path": str(tmp_path / "pred_trace.json"),
        }
    )
    backbone = _FirstFeatureBackbone()
    w = _DummyFitPredictWrapper(
        backbone=backbone,
        yield_strategy=True,
        yield_batch_size=2,
        verbose=True,
        profile_predict_cfg=cfg,
    )
    monkeypatch.setattr(w, "_check_stop_profiler", lambda _c: None)
    X = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = w.predict(X)
    assert out.shape == (2, 1)
    w.profiler.stop()


def test_check_stop_profiler_exports_and_exits(monkeypatch, tmp_path):
    import importlib

    from torch.profiler import ProfilerAction

    cfg = OmegaConf.create({"trace_path": str(tmp_path / "chrome.json")})

    class _FakeProfiler:
        step_num = 0

        def stop(self):
            pass

        def key_averages(self):
            class _KA:
                def table(self, **_kw):
                    return "avg-table"

            return _KA()

        def export_chrome_trace(self, path):
            Path(path).write_text("{}")

    w = _DummyFitPredictWrapper(backbone=_NumpyBackbone())
    w.profiler = _FakeProfiler()
    w.sched = lambda _n: ProfilerAction.NONE

    def _exit_raises(code=0):
        raise SystemExit(code)

    base_mod = importlib.import_module(AbstractFitPredictWrapper.__module__)
    monkeypatch.setattr(base_mod.sys, "exit", _exit_raises)

    with pytest.raises(SystemExit) as excinfo:
        w._check_stop_profiler(cfg)
    assert excinfo.value.code == 0
    assert tmp_path.joinpath("chrome.json").exists()


def test_abstract_fit_predict_super_calls_execute_pass():
    w = _AbstractPassThrough(backbone=_NumpyBackbone())
    assert w.serialize_model("t") is None
    assert w.load_model("t") is None


def test_abstract_allows_multi_target_pass_via_base_property_fget():
    class _Minimal(AbstractFitPredictWrapper):
        @property
        def allows_multi_target(self) -> bool:
            return False

        def serialize_model(self, task_id):
            return None

        def load_model(self, task_id):
            return None

    m = _Minimal(backbone=_NumpyBackbone())
    assert AbstractFitPredictWrapper.allows_multi_target.fget(m) is None


def test_abstract_feed_forward_wrapper_repr():
    bb = _BackboneWithOutChannels()
    w = _ReprFeedForwardWrapper(
        backbone=bb, dimensions=3, residual_connections=False, out_channels=1
    )
    text = repr(w)
    assert "ReprFeedForwardWrapper" in text
    assert "out_channels=7" in text
    assert "dimensions=3" in text
    assert "residual_connections=False" in text


def test_abstract_feed_forward_training_wrapper_repr():
    backbone = torch.nn.Linear(2, 2)
    w = _FFTrainingWrapper(backbone=backbone, out_channels=2, num_cell_dimensions=1)
    assert repr(w) == f"_FFTrainingWrapper(backbone={backbone})"
