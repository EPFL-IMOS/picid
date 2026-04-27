"""Integration tests for Spacetimeformer_Forecaster.

Follows pattern from test_tide_model.py and test_patchtst_model.py.
Tests instantiate Spacetimeformer_Forecaster with minimal config and run
forward_model_pass for regression, classification, and state_forecasting.
"""

import numpy as np
import pytest
import hydra
from hydra.core.global_hydra import GlobalHydra
import torch

from picid.model.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    STATE_FORECASTING_TASKS,
)
from picid.model.forecasters.spacetimeformer_model.spacetimeformer_model import (
    Spacetimeformer_Forecaster,
)
from test.conftest import BATCH_SIZE
from test.utils import ProjectSearchPathPlugin
from typing import Any
from picid.evaluator.base import AbstractEvaluator


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
            "model=stf",
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
    Matches regression / classification path in Spacetimeformer_Forecaster.
    """
    ts = hydra_cfg.model.max_seq_len
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
            "model_cfg": "prognostics/stf.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/stf.yaml",
        },
        {
            "task": "state_forecasting/c4_l0_p1",
            "model_cfg": "state_forecasting/stf.yaml",
        },
    ],
    ids=["rul", "diagnostics", "state_forecasting"],
    indirect=True,
)
def test_initialize_model(hydra_model):
    assert hydra_model.task_type in (
        CLASSIFICATION_TASKS + REGRESSION_TASKS + STATE_FORECASTING_TASKS
    )
    assert isinstance(hydra_model, Spacetimeformer_Forecaster)


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/stf.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/stf.yaml",
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
        return (fake_forecast_out, torch.zeros_like(fake_forecast_out), (None, None))

    def fake_forecasting_loss(outputs, y_t, time_mask=None, feat_mask=None):
        called["time_mask"] = time_mask
        called["feat_mask"] = feat_mask
        return torch.tensor(1.23), torch.ones_like(y_t, dtype=torch.bool)

    monkeypatch.setattr(hydra_model, "forward", fake_forward)
    monkeypatch.setattr(hydra_model, "forecasting_loss", fake_forecasting_loss)

    out = hydra_model.compute_loss(
        batch=test_batch,
        time_mask="tm",
        forward_kwargs={"target_mask": None},
    )

    assert out["forecast_loss"].isclose(torch.tensor(1.23))
    assert out["forecast_out"] is fake_forecast_out
    assert "forecast_mask" in out
    assert called["time_mask"] == "tm"
    assert called["feat_mask"] is None


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/stf.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/stf.yaml",
        },
    ],
    ids=["rul", "diagnostics"],
    indirect=True,
)
def test_step_train_mode(hydra_model, test_batch_reg_class, monkeypatch):
    test_batch = test_batch_reg_class
    fake_out = torch.randn_like(test_batch[-1])

    def fake_compute_loss(batch, time_mask=None, forward_kwargs=None):
        assert time_mask is None
        return {
            "forecast_loss": torch.tensor(0.5),
            "class_loss": torch.tensor(0.0),
            "recon_loss": torch.tensor(0.0),
            "acc": -1.0,
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
            "model_cfg": "prognostics/stf.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/stf.yaml",
        },
        {
            "task": "state_forecasting/c4_l0_p1",
            "model_cfg": "state_forecasting/stf.yaml",
        },
    ],
    ids=["rul", "diagnostics", "state_forecasting"],
    indirect=True,
)
def test_forward_model_including_model(hydra_model, hydra_cfg):
    cfg = hydra_cfg

    batch_size = BATCH_SIZE
    ts = cfg.model.max_seq_len
    d_x = cfg.model.d_x
    d_yt = cfg.model.d_yt

    if hydra_model.task_type in STATE_FORECASTING_TASKS:
        # state_forecasting: x_c=None, y_c=(B, seq+pred, d_yc), x_t=None, y_t=(B, pred, d_yt)
        seq_len = cfg.task_definition.seq_len
        pred_len = cfg.task_definition.pred_len
        y_c = torch.randn(batch_size, seq_len + pred_len, d_yt)
        y_t = torch.randn(batch_size, pred_len, d_yt)
        x_c, x_t = None, None
    else:
        x_t = torch.randn(batch_size, ts, d_x)
        x_c = None
        y_c = None
        # For decoder_only (regression/classification), y_t time dim must be <= x_t.
        # Spacetimeformer slices output to y_t.shape[1]; use 1 for single-step prediction.
        if hydra_model.task_type in REGRESSION_TASKS:
            y_t = torch.randn(batch_size, 1, d_yt)
        elif hydra_model.task_type in CLASSIFICATION_TASKS:
            y_t = torch.randint(0, cfg.task_definition.num_classes, (batch_size, 1, 1))
        else:
            raise ValueError("Unknown task type")

    result = hydra_model.forward_model_pass(
        x_c=x_c,
        y_c=y_c,
        x_t=x_t,
        y_t=y_t,
    )

    forecast_out = result[0]
    assert isinstance(forecast_out, torch.Tensor)

    if hydra_model.task_type in REGRESSION_TASKS:
        assert forecast_out.shape == (batch_size, 1, d_yt)
    elif hydra_model.task_type in CLASSIFICATION_TASKS:
        assert forecast_out.shape == (
            batch_size,
            1,
            cfg.task_definition.num_classes,
        )
    elif hydra_model.task_type in STATE_FORECASTING_TASKS:
        assert forecast_out.shape == (batch_size, pred_len, d_yt)
    else:
        raise ValueError("Unknown task type")


@pytest.mark.parametrize(
    "hydra_cfg",
    [
        {
            "task": "prognostics/rul",
            "model_cfg": "prognostics/stf.yaml",
        },
        {
            "task": "diagnostics/concepts_n_cmapss_multi",
            "model_cfg": "diagnostics/stf.yaml",
        },
        {
            "task": "state_forecasting/c4_l0_p1",
            "model_cfg": "state_forecasting/stf.yaml",
        },
    ],
    ids=["rul", "diagnostics", "state_forecasting"],
    indirect=True,
)
def test_forward_model_pass(hydra_model, hydra_cfg, monkeypatch):
    assert hydra_model.task_type in (
        CLASSIFICATION_TASKS + REGRESSION_TASKS + STATE_FORECASTING_TASKS
    )

    cfg = hydra_cfg
    called = {}

    batch_size = BATCH_SIZE
    ts = cfg.model.max_seq_len
    d_x = cfg.model.d_x
    d_yt = cfg.model.d_yt

    x_c, x_t = None, None
    if hydra_model.task_type in STATE_FORECASTING_TASKS:
        seq_len = cfg.task_definition.seq_len
        pred_len = cfg.task_definition.pred_len
        y_c = torch.randn(batch_size, seq_len + pred_len, d_yt)
        y_t = torch.randn(batch_size, pred_len, d_yt)
        expected_forecast_shape = (batch_size, pred_len, d_yt)
    else:
        x_t = torch.randn(batch_size, ts, d_x)
        y_c = None
        if hydra_model.task_type in REGRESSION_TASKS:
            y_t = torch.randn(batch_size, 1, d_yt)
            expected_forecast_shape = (batch_size, 1, d_yt)
        elif hydra_model.task_type in CLASSIFICATION_TASKS:
            y_t = torch.randint(0, cfg.task_definition.num_classes, (batch_size, 1, 1))
            expected_forecast_shape = (batch_size, 1, cfg.task_definition.num_classes)
        else:
            raise ValueError("Unknown task type")

    def fake_forward(enc_x, enc_y, dec_x, dec_y, output_attention=False):
        called["enc_x"] = enc_x
        called["enc_y"] = enc_y
        called["dec_x"] = dec_x
        called["dec_y"] = dec_y
        if hydra_model.task_type in REGRESSION_TASKS:
            return (
                torch.randn(batch_size, 1, d_yt),
                None,
                (None, None),
                None,
                (None, None),
            )
        elif hydra_model.task_type in CLASSIFICATION_TASKS:
            return (
                torch.randn(batch_size, 1, cfg.task_definition.num_classes),
                None,
                (None, None),
                None,
                (None, None),
            )
        elif hydra_model.task_type in STATE_FORECASTING_TASKS:
            return (
                torch.randn(batch_size, pred_len, d_yt),
                None,
                (None, None),
                None,
                (None, None),
            )
        else:
            raise ValueError("Unknown task type")

    monkeypatch.setattr(hydra_model.spacetimeformer, "forward", fake_forward)

    result = hydra_model.forward_model_pass(
        x_c=x_c,
        y_c=y_c,
        x_t=x_t,
        y_t=y_t,
    )

    forecast_out = result[0]
    assert forecast_out.shape == expected_forecast_shape


def test_spacetimeformer_init_negative_start_token_builds_router():
    evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
    model = Spacetimeformer_Forecaster(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.001),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        ),
        d_yc=2,
        d_yt=1,
        d_x=4,
        task_type="forecasting",
        n_timefeatures=2,
        start_token_len=-2,
        max_seq_len=32,
        d_model=16,
        d_queries_keys=8,
        d_values=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        time_emb_dim=2,
        timetable_emb_type="linear",
        evaluators=evaluators,
    )

    assert model.start_token_len == -2
    assert model.d_x_embedding_router is not None
    assert torch.equal(model.d_x_embedding_router.cpu(), torch.tensor([0, 0, 1, 1]))


def test_spacetimeformer_forward_model_pass_masks_decoder_only_target_and_returns_attn(
    monkeypatch,
):
    evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
    model = Spacetimeformer_Forecaster(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.001),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        ),
        d_yc=0,
        d_yt=1,
        d_x=4,
        task_type="rul",
        max_seq_len=24,
        d_model=16,
        d_queries_keys=8,
        d_values=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        evaluators=evaluators,
    )
    captured = {}

    def fake_forward(enc_x, enc_y, dec_x, dec_y, output_attention=False):
        captured["enc_x"] = enc_x
        captured["enc_y"] = enc_y
        captured["dec_x"] = dec_x
        captured["dec_y"] = dec_y
        captured["output_attention"] = output_attention
        return (
            torch.randn(2, dec_y.shape[1], 1),
            torch.randn(2, dec_y.shape[1], 1),
            (None, None),
            "attn-map",
            (None, None),
        )

    monkeypatch.setattr(model.spacetimeformer, "forward", fake_forward)

    x_t = torch.randn(2, 5, 4)
    y_t = torch.randn(2, 5)

    forecast_out, recon_out, _, attn = model.forward_model_pass(
        x_c=None,
        y_c=None,
        x_t=x_t,
        y_t=y_t,
        output_attn=True,
        target_mask=[0],
    )

    assert forecast_out.shape == (2, 5, 1)
    assert recon_out.shape == (2, 5, 1)
    assert attn == "attn-map"
    assert captured["enc_x"] is None
    assert captured["enc_y"] is None
    assert captured["output_attention"] is True
    assert torch.equal(captured["dec_y"][:, :, 0], torch.zeros(2, 5))


def test_spacetimeformer_forward_model_pass_masks_2d_context_in_state_forecasting(
    monkeypatch,
):
    evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
    model = Spacetimeformer_Forecaster(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.001),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        ),
        d_yc=1,
        d_yt=1,
        d_x=0,
        task_type="state_forecasting",
        max_seq_len=24,
        d_model=16,
        d_queries_keys=8,
        d_values=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        mask_y_c=True,
        evaluators=evaluators,
    )
    captured = {}

    def fake_forward(enc_x, enc_y, dec_x, dec_y, output_attention=False):
        captured["enc_x"] = enc_x
        captured["enc_y"] = enc_y
        captured["dec_x"] = dec_x
        captured["dec_y"] = dec_y
        return (
            torch.randn(2, dec_y.shape[1], 1),
            torch.randn(2, enc_y.shape[1], 1),
            (None, None),
            None,
            (None, None),
        )

    monkeypatch.setattr(model.spacetimeformer, "forward", fake_forward)

    y_c = torch.randn(2, 6)
    y_t = torch.randn(2, 3)

    forecast_out, _, _ = model.forward_model_pass(x_c=None, y_c=y_c, x_t=None, y_t=y_t)

    assert forecast_out.shape[0] == 2
    assert forecast_out.shape[2] == 1
    assert captured["enc_y"].shape == (2, 6, 1)
    assert torch.equal(captured["enc_y"], torch.zeros(2, 6, 1))
    assert captured["enc_x"].shape == (2, 6, 1)
    assert captured["dec_x"].shape[0] == 2
    assert captured["dec_x"].shape[2] == 1


def test_spacetimeformer_compute_loss_handles_target_recon_and_class_terms(monkeypatch):
    evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
    model = Spacetimeformer_Forecaster(
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=0.001),
        scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        ),
        d_yc=2,
        d_yt=2,
        d_x=2,
        task_type="forecasting",
        max_seq_len=24,
        d_model=16,
        d_queries_keys=8,
        d_values=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        recon_loss_imp=1.0,
        class_loss_imp=1.0,
        embed_method="spatio-temporal",
        evaluators=evaluators,
    )

    def fake_forward(x_c, y_c, x_t, y_t, **forward_kwargs):
        return (
            torch.randn(2, 4, 2),
            torch.randn(2, 6, 2),
            (torch.randn(8, 3), torch.randint(0, 3, (8,))),
        )

    forecast_calls = []

    def fake_forecasting_loss(outputs, y_t, time_mask=None, feat_mask=None):
        forecast_calls.append((outputs, y_t, time_mask))
        return torch.tensor(0.5), torch.ones_like(y_t, dtype=torch.bool)

    monkeypatch.setattr(model, "forward", fake_forward)
    monkeypatch.setattr(model, "forecasting_loss", fake_forecasting_loss)
    monkeypatch.setattr(
        model, "classification_loss", lambda logits, labels: (torch.tensor(0.7), 0.8)
    )

    batch = (
        torch.randn(2, 6, 2),
        torch.randn(2, 6, 2),
        torch.randn(2, 4, 2),
        torch.randn(2, 4, 2),
    )
    loss_dict = model.compute_loss(
        batch=batch, time_mask=2, forward_kwargs={"target_mask": [0]}
    )

    assert loss_dict["forecast_loss"].equal(torch.tensor(0.5))
    assert loss_dict["recon_loss"].equal(torch.tensor(0.5))
    assert loss_dict["class_loss"].equal(torch.tensor(0.7))
    assert loss_dict["acc"] == 0.8
    assert forecast_calls[0][1].shape == (2, 4, 1)
    assert forecast_calls[1][1].shape == (2, 6, 2)
