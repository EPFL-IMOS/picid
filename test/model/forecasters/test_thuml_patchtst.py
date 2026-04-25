"""Unit tests for PatchTST Model, Transpose, FlattenHead (thuml_patchtst.py)."""

import torch
from types import SimpleNamespace

from picid.model.forecasters.patchtst_model.thuml_patchtst import (
    FlattenHead,
    Model,
    Transpose,
)


def _patchtst_config(task_name="long_term_forecast", seq_len=96, pred_len=24, enc_in=1):
    return SimpleNamespace(
        task_name=task_name,
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=enc_in,
        num_class=2 if task_name == "classification" else None,
        num_reg_targets=1 if task_name == "regression" else None,
        d_model=16,
        dropout=0.1,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        n_heads=2,
        activation="gelu",
        factor=3,
    )


class TestTranspose:
    def test_forward(self):
        t = Transpose(0, 1)
        x = torch.randn(2, 3)
        out = t(x)
        assert out.shape == (3, 2)

    def test_forward_contiguous(self):
        t = Transpose(0, 1, contiguous=True)
        x = torch.randn(2, 3)
        out = t(x)
        assert out.is_contiguous()


class TestFlattenHead:
    def test_forward_shape(self):
        h = FlattenHead(
            nf=160, target_window=24, head_dropout=0
        )  # nf = d_model * patch_num
        x = torch.randn(2, 4, 16, 10)  # bs, nvars, d_model, patch_num
        out = h(x)
        assert out.shape == (2, 4, 24)


class TestPatchTSTModel:
    def test_init_forecast(self):
        cfg = _patchtst_config()
        m = Model(cfg, patch_len=16, stride=8)
        assert m.task_name == "long_term_forecast"
        assert m.head is not None

    def test_forward_forecast_shape(self):
        cfg = _patchtst_config(seq_len=96, pred_len=24)
        m = Model(cfg, patch_len=16, stride=8)
        B, L, D = 2, 96, 1
        x = torch.randn(B, L, D)
        out = m(x, None, None, None)
        assert out.shape == (B, 24, D)

    def test_forward_regression_shape(self):
        cfg = _patchtst_config(task_name="regression", seq_len=96, enc_in=4)
        m = Model(cfg, patch_len=16, stride=8)
        B, L, D = 2, 96, 4
        x = torch.randn(B, L, D)
        out = m(x, None, None, None)
        assert out.shape == (B, 1)

    def test_forward_classification_shape(self):
        cfg = _patchtst_config(task_name="classification", seq_len=96, enc_in=4)
        m = Model(cfg, patch_len=16, stride=8)
        B, L, D = 2, 96, 4
        x = torch.randn(B, L, D)
        out = m(x, None, None, None)
        assert out.shape == (B, 2)
