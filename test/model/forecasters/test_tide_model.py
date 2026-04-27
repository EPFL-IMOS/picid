import numpy as np

import pytest
import hydra
import torch

from picid.model.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS
from picid.model.forecasters.tide_model.tide_model import TiDE_Forecaster
from test.conftest import BATCH_SIZE
from test.utils import ProjectSearchPathPlugin
from typing import Any
from picid.evaluator.base import AbstractEvaluator
from hydra.core.global_hydra import GlobalHydra


ProjectSearchPathPlugin.register()


class MockEvaluator(AbstractEvaluator):
    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def reset(self) -> None:
        self.results.clear()

    def compute(self, mode, epoch, step):
        return self.results

    def update(self, model_outs):
        self.results = {
            "loss": 0.0,
        }
        return self.results


# ----------------------
# Fixture
# ----------------------


@pytest.fixture(scope="module", autouse=True)
def hydra_context(request):
    # Ensure any previous initialization is cleared before starting
    GlobalHydra.instance().clear()
    with hydra.initialize(version_base="1.3", config_path="configs", job_name="run"):
        yield


@pytest.fixture(scope="module")
def path_cfg(request) -> str:
    """Return the path to the configuration file."""
    return request.config.getoption("--paths")


@pytest.fixture(scope="module")
def hydra_cfg(request, path_cfg):
    params = request.param
    task = params["task"]
    model_cfg = params["model_cfg"]

    cfg = hydra.compose(
        config_name="test.yaml",
        overrides=[
            "model=tide",
            f"task_definition={task}",
            f"+model_configs={model_cfg}",
            f"paths={path_cfg}",
        ],
        return_hydra_config=True,
    )
    return cfg


@pytest.fixture(scope="module")
def hydra_model(hydra_cfg):
    cfg = hydra_cfg

    optimizer_factory = hydra.utils.instantiate(
        cfg.optimization.optimizer,
        _partial_=True,
    )

    scheduler_factory = None
    if getattr(cfg.optimization, "scheduler", None) is not None:
        scheduler_factory = hydra.utils.instantiate(
            cfg.optimization.scheduler,
            _partial_=True,
        )

    evaluators = {split: MockEvaluator() for split in ("train", "val", "test")}

    model = hydra.utils.instantiate(
        cfg.model,
        evaluators=evaluators,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )

    return model


@pytest.fixture
def test_batch_reg_class(hydra_cfg):
    """
    Batch is a 4-tuple: (x_c, y_c, x_t, y_t)
    Matches regression / classification path in TiDE_Forecaster.
    """
    ts = hydra_cfg.model.ts_in
    d_x = hydra_cfg.model.d_x
    d_yt = hydra_cfg.model.d_yt

    x_c = None
    y_c = None

    x_t = torch.randn(BATCH_SIZE, ts, d_x)
    y_t = torch.randn(BATCH_SIZE, ts, d_yt)

    return x_c, y_c, x_t, y_t


# ----------------------
# Tests
# ----------------------


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/tide.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/tide.yaml",
        },
    ],
    ids=["rul", "diagnostics"],
    indirect=True,
)
def test_initialize_model(hydra_model):
    assert hydra_model.task_type in CLASSIFICATION_TASKS + REGRESSION_TASKS
    assert isinstance(hydra_model, TiDE_Forecaster)


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/tide.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/tide.yaml",
        },
    ],
    ids=["rul", "diagnostics"],
    indirect=True,
)
def test_compute_loss_calls_forecasting_loss(
    hydra_model, test_batch_reg_class, monkeypatch
):
    test_batch = test_batch_reg_class
    called = {}

    fake_forecast_out = torch.randn_like(test_batch[-1])

    def fake_forward(x_c, y_c, x_t, y_t, **kwargs):
        called["forward_kwargs"] = kwargs
        return (fake_forecast_out,)

    def fake_forecasting_loss(outputs, y_t, time_mask=None, feat_mask=None):
        called["time_mask"] = time_mask
        called["feat_mask"] = feat_mask
        return torch.tensor(1.23), torch.ones_like(y_t, dtype=torch.bool)

    monkeypatch.setattr(hydra_model, "forward", fake_forward)
    monkeypatch.setattr(hydra_model, "forecasting_loss", fake_forecasting_loss)

    out = hydra_model.compute_loss(
        batch=test_batch,
        time_mask="tm",
        forward_kwargs={"target_mask": "fm"},
    )

    assert out["forecast_loss"].isclose(torch.tensor(1.23))
    assert out["forecast_out"] is fake_forecast_out
    assert "forecast_mask" in out
    assert called["time_mask"] == "tm"
    assert called["feat_mask"] == "fm"


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/tide.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/tide.yaml",
        },
    ],
    ids=["rul", "diagnostics"],
    indirect=True,
)
def test_step_train_mode(hydra_model, test_batch_reg_class, monkeypatch):
    test_batch = test_batch_reg_class
    fake_out = torch.randn_like(test_batch[-1])  # (x_c, y_c, x_t, y_t)

    def fake_compute_loss(batch, time_mask=None, forward_kwargs=None):
        assert time_mask is None  # We do not use time masking features for now
        return {
            "forecast_loss": torch.tensor(0.5),
            "forecast_out": fake_out,
            "forecast_mask": torch.ones_like(test_batch[-1], dtype=torch.bool),
        }

    monkeypatch.setattr(hydra_model, "compute_loss", fake_compute_loss)

    stats = hydra_model.step(batch=test_batch, train=True)

    assert stats["loss"] == torch.tensor(0.5)
    assert stats["forecast_loss"] == torch.tensor(0.5)
    assert stats["predictions"].shape == fake_out.cpu().numpy().shape
    assert stats["targets"].shape == test_batch[-1].cpu().numpy().shape

    assert isinstance(stats["predictions"], np.ndarray)
    assert isinstance(stats["targets"], np.ndarray)


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/tide.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/tide.yaml",
        },
    ],
    ids=["rul", "diagnostics"],
    indirect=True,
)
def test_forward_model_including_model(hydra_model, hydra_cfg):
    cfg = hydra_cfg

    batch_size = BATCH_SIZE
    ts = cfg.model.ts_in
    d_x = cfg.model.d_x

    x_t = torch.randn(batch_size, ts, d_x)

    if hydra_model.task_type in REGRESSION_TASKS:
        y_t = torch.randn(batch_size, ts, 1)
    elif hydra_model.task_type in CLASSIFICATION_TASKS:
        y_t = torch.randint(0, 2, (batch_size, ts, 1))
    else:
        raise ValueError("Unknown task type")

    (out,) = hydra_model.forward_model_pass(
        x_c=None,
        y_c=None,
        x_t=x_t,
        y_t=y_t,
    )

    assert isinstance(out, torch.Tensor)

    if hydra_model.task_type in REGRESSION_TASKS:
        assert out.shape == (batch_size, 1, 1)
    elif hydra_model.task_type in CLASSIFICATION_TASKS:
        assert out.shape == (batch_size, 1, cfg.task_definition.num_classes)
    else:
        raise ValueError("Unknown task type")


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/tide.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/tide.yaml",
        },
    ],
    ids=["rul", "diagnostics"],
    indirect=True,
)
def test_forward_model_pass(hydra_model, hydra_cfg, monkeypatch):
    assert hydra_model.task_type in CLASSIFICATION_TASKS + REGRESSION_TASKS
    assert hydra_model.include_x_t is True

    cfg = hydra_cfg
    called = {}

    def fake_forward(x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None):
        assert mask is None
        called["x_enc"] = x_enc
        called["x_mark_enc"] = x_mark_enc
        called["x_dec"] = x_dec
        called["batch_y_mark"] = batch_y_mark

        if hydra_model.task_type in REGRESSION_TASKS:
            return torch.randn(batch_size, 1, 1)
        elif hydra_model.task_type in CLASSIFICATION_TASKS:
            return torch.randn(batch_size, 1, cfg.task_definition.num_classes)
        else:
            raise ValueError("Unknown task type")

    monkeypatch.setattr(hydra_model.model, "forward", fake_forward)

    batch_size = BATCH_SIZE
    ts = cfg.model.ts_in
    d_x = cfg.model.d_x

    x_t = torch.randn(batch_size, ts, d_x)

    if hydra_model.task_type in REGRESSION_TASKS:
        assert hydra_model.model.task_name == "regression"
        y_t = torch.randn(batch_size, d_x)
    elif hydra_model.task_type in CLASSIFICATION_TASKS:
        assert hydra_model.model.task_name == "classification"
        y_t = torch.randint(0, 2, (batch_size, 1))
    else:
        raise ValueError("Unknown task type")

    (out,) = hydra_model.forward_model_pass(
        x_c=None,
        y_c=None,
        x_t=x_t,
        y_t=y_t,
    )

    assert called["x_enc"].shape == (batch_size, ts, 1)
    assert called["batch_y_mark"].shape == (batch_size, ts, d_x)
    assert called["x_mark_enc"].shape == (batch_size, ts, d_x)
    assert called["x_dec"] is None

    if hydra_model.task_type in REGRESSION_TASKS:
        assert out.shape == (batch_size, 1, 1)
    elif hydra_model.task_type in CLASSIFICATION_TASKS:
        assert out.shape == (batch_size, 1, cfg.task_definition.num_classes)
    else:
        raise ValueError("Unknown task type")
