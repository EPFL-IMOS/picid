"""Unit tests for TiDE Model (tide.py)."""

import torch
from omegaconf import OmegaConf

from picid.model.forecasters.tide_model.tide import Model as TiDE


def _tide_config(task_name="long_term_forecast", seq_len=96, pred_len=24):
    """Config for TiDE: freq or feature_dim must be None (assert in Model)."""
    cfg = {
        "task_name": task_name,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "freq": "h" if task_name != "state_forecasting" else None,
        "feature_dim": None if task_name != "state_forecasting" else 4,
        "feature_encode_dim": 4,
        "d_model": 16,
        "e_layers": 1,
        "d_layers": 1,
        "d_ff": 32,
        "dropout": 0.1,
        "c_out": 1,
        "num_class": 2 if task_name == "classification" else None,
    }
    return OmegaConf.create(cfg)


def test_tide_forward_forecast_shape():
    cfg = _tide_config()
    m = TiDE(cfg)
    B, L, D = 2, 96, 1
    x_enc = torch.randn(B, L, D)
    # x_mark_enc: encoder marks (B, L, feature_dim); batch_y_mark: full (B, L+pred_len, feature_dim)
    # forward concats [x_mark_enc, batch_y_mark[:, -pred_len:]] -> (B, L+pred_len, feature_dim)
    x_mark_enc = torch.randn(B, L, 4)
    batch_y_mark = torch.randn(B, L + 24, 4)
    out = m(x_enc, x_mark_enc, None, batch_y_mark)
    assert out.shape == (B, 24, D)


def test_tide_forward_state_forecasting_shape():
    cfg = _tide_config(task_name="state_forecasting")
    cfg.c_out = 2
    m = TiDE(cfg)
    B, L, D = 2, 96, 2
    x_enc = torch.randn(B, L, D)
    out = m(x_enc, None, None, None)
    assert out.shape == (B, 24, D)
