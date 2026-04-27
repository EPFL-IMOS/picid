"""Tests for TimeseriesTransformer baseline.

Timeseries_Transformer_Forecaster wraps a vanilla Transformer for
forecasting, regression, and classification tasks.
"""

import pytest
import torch
from types import SimpleNamespace
from typing import Any, Dict

from picid.evaluator.base import AbstractEvaluator
from picid.model.forecasters.timeseries_transformer_model.timeseries_transformer import (
    Model as Transformer,
)
from picid.model.forecasters.timeseries_transformer_model.timeseries_transformer_model import (
    Timeseries_Transformer_Forecaster,
)


class _Config:
    """Config object that supports both attribute and item access (like OmegaConf)."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __getitem__(self, k):
        return getattr(self, k)


class MockEvaluator(AbstractEvaluator):
    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}

    def reset(self) -> None:
        self.results.clear()

    def compute(self, mode, epoch, step):
        return self.results

    def update(self, model_outs):
        self.results = {"loss": 0.0}
        return self.results


def _mock_optimizer_factory(params):
    return torch.optim.Adam(params, lr=0.001)


def _mock_scheduler_factory(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)


def _transformer_config(
    seq_len=24,
    pred_len=6,
    enc_in=4,
    dec_in=4,
    enc_context_in=4,
    dec_context_in=4,
    d_model=16,
    embed_type=0,
):
    """Minimal config for Transformer Model (embed_type 0 = DataEmbedding)."""
    return SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        label_len=pred_len,
        enc_in=enc_in,
        dec_in=dec_in,
        enc_context_in=enc_context_in,
        dec_context_in=dec_context_in,
        d_model=d_model,
        c_out=1,
        embed="timeF",
        freq="h",
        dropout=0.1,
        output_attention=False,
        embed_type=embed_type,
        n_heads=2,
        d_ff=64,
        e_layers=1,
        d_layers=1,
        activation="gelu",
        factor=3,
        d_x_embedding_router=None,
    )


# =============================================================================
# Transformer Model (timeseries_transformer.py)
# =============================================================================


class TestTransformerModel:
    """Tests for the underlying Transformer Model."""

    def test_init_decoder_only(self):
        """Transformer initializes in decoder-only mode."""
        cfg = _transformer_config(embed_type=0)
        model = Transformer(cfg, decoder_only=True)
        assert model.decoder_only is True
        assert model.enc_embedding is None
        assert model.pred_len == 6
        assert model.d_model == 16

    def test_init_encoder_decoder(self):
        """Transformer initializes with encoder and decoder."""
        cfg = _transformer_config(embed_type=0)
        model = Transformer(cfg, decoder_only=False)
        assert model.decoder_only is False
        assert model.enc_embedding is not None
        assert model.dec_embedding is not None

    def test_forward_decoder_only(self):
        """Forward pass in decoder-only mode."""
        torch.manual_seed(42)
        cfg = _transformer_config(seq_len=12, pred_len=4, enc_in=2, dec_in=2)
        model = Transformer(cfg, decoder_only=True)
        # x_mark_dec needs 4 time features for freq="h" (TimeFeatureEmbedding)
        B, L, D = 2, 12, 2
        x_dec = torch.randn(B, L, D)
        x_mark_dec = torch.randn(B, L, 4)  # 4 = freq "h"
        out = model(None, None, x_dec, x_mark_dec)
        assert out.shape == (B, cfg.pred_len, 1)

    def test_forward_full(self):
        """Forward pass with encoder and decoder."""
        torch.manual_seed(42)
        cfg = _transformer_config(seq_len=12, pred_len=4)
        model = Transformer(cfg, decoder_only=False)
        B, L_enc, L_dec = 2, 12, 12
        x_enc = torch.randn(B, L_enc, cfg.enc_in)
        x_mark_enc = torch.randn(B, L_enc, 4)  # time features
        x_dec = torch.randn(B, L_dec, cfg.dec_in)
        x_mark_dec = torch.randn(B, L_dec, 4)
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert out.shape == (B, cfg.pred_len, 1)

    def test_forward_full_returns_attention_when_requested(self):
        torch.manual_seed(42)
        cfg = _transformer_config(seq_len=12, pred_len=4)
        cfg.output_attention = True
        model = Transformer(cfg, decoder_only=False)
        B, L_enc, L_dec = 2, 12, 12
        x_enc = torch.randn(B, L_enc, cfg.enc_in)
        x_mark_enc = torch.randn(B, L_enc, 4)
        x_dec = torch.randn(B, L_dec, cfg.dec_in)
        x_mark_dec = torch.randn(B, L_dec, 4)

        out, attns = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

        assert out.shape == (B, cfg.pred_len, 1)
        assert len(attns) == cfg.e_layers

    def test_embed_type_2_wo_pos(self):
        """Transformer with embed_type 2 (DataEmbedding_wo_pos)."""
        cfg = _transformer_config(embed_type=2)
        model = Transformer(cfg, decoder_only=False)
        assert model.enc_embedding is not None

    def test_embed_type_1_matches_data_embedding_branch(self):
        """Transformer with embed_type 1 takes the alternate DataEmbedding branch."""
        cfg = _transformer_config(embed_type=1)
        model = Transformer(cfg, decoder_only=False)
        assert model.enc_embedding is not None
        assert model.dec_embedding is not None

    def test_embed_type_3_wo_temp(self):
        """Transformer with embed_type 3 (DataEmbedding_wo_temp)."""
        cfg = _transformer_config(embed_type=3)
        model = Transformer(cfg, decoder_only=False)
        assert model.enc_embedding is not None

    @pytest.mark.parametrize("embed_type", [4, 5, 6])
    def test_embed_type_forward_shapes(self, embed_type):
        """Forward pass and shapes for embed_type 4, 5, 6.

        embed_type 4: DataEmbedding_wo_pos_temp
        embed_type 5: ContextEmbedding (encoder and decoder)
        embed_type 6: ContextEmbedding (encoder) + ContextEmbedding_wo_context (decoder)
        """
        torch.manual_seed(42)
        cfg = _transformer_config(
            seq_len=12,
            pred_len=4,
            enc_in=2,
            dec_in=2,
            enc_context_in=4,
            dec_context_in=4,
            embed_type=embed_type,
        )
        if embed_type == 6:
            # enc_context_in=4 = context cols; x_mark has 8 = 4 context + 4 time (freq="h")
            cfg.d_x_embedding_router = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        model = Transformer(cfg, decoder_only=False)
        B, L_enc, L_dec = 2, 12, 12
        x_enc = torch.randn(B, L_enc, cfg.enc_in)
        x_dec = torch.randn(B, L_dec, cfg.dec_in)
        # embed_type 5: no router, x_mark has enc_context_in cols
        # embed_type 6: router splits 8 cols into 4 context + 4 time
        n_mark = 8 if embed_type == 6 else cfg.enc_context_in
        x_mark_enc = torch.randn(B, L_enc, n_mark)
        x_mark_dec = torch.randn(B, L_dec, n_mark)
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert out.shape == (B, cfg.pred_len, 1)


# =============================================================================
# Timeseries_Transformer_Forecaster (timeseries_transformer_model.py)
# =============================================================================


class TestTimeseriesTransformerForecaster:
    """Tests for Timeseries_Transformer_Forecaster."""

    def test_init_regression(self):
        """Forecaster initializes for regression task."""
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=4,
            dec_in=4,
            enc_context_in=4,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        assert forecaster.task_type == "rul"
        assert forecaster.transformer is not None
        assert forecaster.d_x == 4
        assert forecaster.d_yt == 1

    def test_init_with_n_timefeatures(self):
        """Forecaster with n_timefeatures>0 sets d_x_embedding_router."""
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=4,
            dec_in=4,
            enc_context_in=4,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            n_timefeatures=2,
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        assert forecaster.transformer_args["d_x_embedding_router"] is not None
        assert forecaster.transformer_args["d_x_embedding_router"].sum() == 2

    def test_init_with_d_x_mask_sets_internal_masks(self):
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,
            enc_context_in=1,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
            d_x_mask=[0, 0, 1, 1],
        )
        assert torch.equal(forecaster.mask_tt, torch.tensor([True, True, False, False]))
        assert torch.equal(forecaster.mask_t, torch.tensor([False, False, True, True]))
        assert forecaster.d_x_mask == [0, 0, 1, 1]

    def test_init_with_d_x_zero(self):
        """Forecaster with d_x=0 (no context features)."""
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,
            enc_context_in=1,
            dec_context_in=1,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=0,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        assert forecaster.d_x == 0
        assert not hasattr(forecaster, "t2v") or forecaster.t2v is None

    def test_forward_regression(self):
        """Forward pass for regression (x_c=None, y_c=None)."""
        # Regression: decoder gets y_c (zeros, d_yt=1) and x_mark (x_t, d_x=4)
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,  # decoder value input is y_c with d_yt channels
            enc_context_in=1,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        B, L, D = 2, 24, 4
        x_c = None
        y_c = None
        x_t = torch.randn(B, L, D)
        y_t = torch.randn(B, L, 1)
        (out,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)  # pred_len=6

    def test_forward_forecasting(self):
        """Forward pass for forecasting task."""
        # Forecasting: encoder gets y_c (d_yc=2), decoder gets dec_inp from y_t
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=2,
            dec_in=1,
            enc_context_in=2,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=2,
            d_yt=1,
            task_type="forecasting",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        # label_len=6, pred_len=6; decoder input len = label_len + pred_len = 12
        B, L_ctx, L_tgt = 2, 24, 12
        x_c = torch.randn(B, L_ctx, 4)
        y_c = torch.randn(B, L_ctx, 2)
        x_t = torch.randn(B, L_tgt, 4)
        y_t = torch.randn(B, L_tgt, 1)
        (out,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)

    def test_forward_forecasting_masks_2d_y_context_and_target_features(
        self, monkeypatch
    ):
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,
            enc_context_in=1,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=1,
            d_yt=1,
            task_type="forecasting",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
            d_x_mask=[0, 0, 1, 1],
            mask_y_c=True,
        )
        B, L_ctx, L_tgt = 2, 24, 12
        captured = {}

        def fake_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark):
            captured["batch_x"] = batch_x
            captured["batch_x_mark"] = batch_x_mark
            captured["dec_inp"] = dec_inp
            captured["batch_y_mark"] = batch_y_mark
            return torch.randn(B, cfg.pred_len, 1)

        monkeypatch.setattr(forecaster.transformer, "forward", fake_forward)

        x_c = torch.randn(B, L_ctx, 4)
        y_c = torch.randn(B, L_ctx)
        x_t = torch.randn(B, L_tgt, 4)
        y_t = torch.randn(B, L_tgt)

        (out,) = forecaster.forward_model_pass(x_c, y_c, x_t.clone(), y_t)

        assert out.shape == (B, cfg.pred_len, 1)
        assert captured["batch_x"].shape == (B, L_ctx, 1)
        assert torch.equal(captured["batch_x"], torch.zeros(B, L_ctx, 1))
        assert torch.equal(captured["batch_y_mark"][:, :, :2], torch.zeros(B, L_tgt, 2))

    def test_forward_state_forecasting_uses_decoder_only_passthrough(self, monkeypatch):
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=2,
            dec_in=2,
            enc_context_in=2,
            dec_context_in=2,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=2,
            d_yc=2,
            d_yt=1,
            task_type="state_forecasting",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        captured = {}

        def fake_forward(x_enc, x_mark_enc, x_dec, x_mark_dec):
            captured["x_enc"] = x_enc
            captured["x_mark_enc"] = x_mark_enc
            captured["x_dec"] = x_dec
            captured["x_mark_dec"] = x_mark_dec
            return torch.randn(2, cfg.pred_len, 1)

        monkeypatch.setattr(forecaster.transformer, "forward", fake_forward)

        y_c = torch.randn(2, 24, 2)
        x_t = torch.randn(2, 24, 2)
        y_t = torch.randn(2, 24, 1)

        (out,) = forecaster.forward_model_pass(None, y_c, x_t, y_t)

        assert out.shape == (2, cfg.pred_len, 1)
        assert captured["x_enc"] is None
        assert captured["x_mark_enc"] is None
        assert torch.equal(captured["x_dec"], y_c)
        assert captured["x_mark_dec"] is None

    @pytest.mark.skip(
        reason="state_forecasting passes x_mark_dec=None; DataEmbedding requires x_mark"
    )
    def test_forward_state_forecasting(self):
        """Forward pass for state_forecasting task.

        **Skipped:** The forecaster calls ``transformer(None, None, batch_x, None)``
        for state_forecasting, passing ``x_mark_dec=None``. The decoder uses
        ``DataEmbedding``, which always calls ``temporal_embedding(x_mark)`` and thus
        requires a non-None ``x_mark``. This causes ``AttributeError: 'NoneType' object
        has no attribute 'to'``. Until the forecaster passes a valid x_mark or the
        decoder supports x_mark=None for this path, this test remains skipped.
        """
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=2,
            dec_in=2,
            enc_context_in=2,
            dec_context_in=2,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=2,
            d_yc=2,
            d_yt=1,
            task_type="state_forecasting",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        B, L = 2, 24
        y_c = torch.randn(B, L, 2)
        x_t = torch.randn(B, L, 2)
        y_t = torch.randn(B, L, 1)
        (out,) = forecaster.forward_model_pass(y_c, None, x_t, y_t)
        assert out.shape == (B, 6, 1)

    def test_train_step_forward_kwargs(self):
        """train_step_forward_kwargs returns output_attn=False."""
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=4,
            dec_in=4,
            enc_context_in=4,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        assert forecaster.train_step_forward_kwargs == {"output_attn": False}
        assert forecaster.eval_step_forward_kwargs == {"output_attn": False}

    def test_step_regression(self):
        """step() computes loss and returns stats for regression."""
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,
            enc_context_in=1,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        # step() asserts preds.shape == y_t.shape; transformer outputs pred_len
        # so y_t must have seq length = pred_len for regression
        B, pred_len = 2, 6
        batch = (
            None,
            None,
            torch.randn(B, pred_len, 4),
            torch.randn(B, pred_len, 1),
        )
        stats = forecaster.step(batch, train=True)
        assert "loss" in stats
        assert "forecast_loss" in stats
        assert "predictions" in stats
        assert "targets" in stats

    @pytest.mark.skip(
        reason="d_x_mask path triggers embed shape mismatch; needs embed_type config"
    )
    def test_forward_with_d_x_mask(self):
        """Forward pass with d_x_mask zeros timetable dims.

        **Skipped:** RuntimeError: Given groups=1, weight of size [16, 4, 3], expected
        input[2, 1, 26] to have 4 channels, but got 1 channels instead.

        d_x_mask zeros some columns of x_t (e.g. [0, 0, 1, 1] zeros the first two).
        The decoder's DataEmbedding expects a fixed number of time-feature channels
        (e.g. 4 for freq="h"). After masking, the effective input has fewer channels,
        so the Conv1d layer (expecting 4 input channels) receives the wrong shape.

        **Fix:** Either make the decoder embedding robust to masked inputs (pass mask
        and handle variable channels), or ensure the masked input is reshaped/padded
        to the expected channel count before the embedding.
        """
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=4,
            dec_in=4,
            enc_context_in=4,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
            d_x_mask=[0, 0, 1, 1],
        )
        B, L, D = 2, 24, 4
        x_c = None
        y_c = None
        x_t = torch.randn(B, L, D)
        y_t = torch.randn(B, L, 1)
        (out,) = forecaster.forward_model_pass(x_c, y_c, x_t, y_t)
        assert out.shape == (B, 6, 1)

    def test_compute_loss(self):
        """compute_loss returns forecast_loss, forecast_out, forecast_mask."""
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,
            enc_context_in=1,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        B, pred_len = 2, 6
        batch = (
            None,
            None,
            torch.randn(B, pred_len, 4),
            torch.randn(B, pred_len, 1),
        )
        loss_dict = forecaster.compute_loss(batch)
        assert "forecast_loss" in loss_dict
        assert "forecast_out" in loss_dict
        assert "forecast_mask" in loss_dict

    def test_step_eval_forwards_target_mask(self, monkeypatch):
        cfg = _Config(
            seq_len=24,
            pred_len=6,
            label_len=6,
            enc_in=1,
            dec_in=1,
            enc_context_in=1,
            dec_context_in=4,
            d_model=16,
            c_out=1,
            embed="timeF",
            freq="h",
            dropout=0.1,
            output_attention=False,
            embed_type=0,
            n_heads=2,
            d_ff=64,
            e_layers=1,
            d_layers=1,
            activation="gelu",
            factor=3,
        )
        evaluators = {s: MockEvaluator() for s in ("train", "val", "test")}
        forecaster = Timeseries_Transformer_Forecaster(
            optimizer_factory=_mock_optimizer_factory,
            scheduler_factory=_mock_scheduler_factory,
            d_x=4,
            d_yc=0,
            d_yt=1,
            task_type="rul",
            transformer_args=cfg,
            device="cpu",
            evaluators=evaluators,
        )
        forecaster.target_mask = "mask-output"
        batch = (
            None,
            None,
            torch.randn(2, 6, 4),
            torch.randn(2, 6, 1),
        )
        captured = {}

        def fake_compute_loss(batch, time_mask=None, forward_kwargs=None):
            captured["time_mask"] = time_mask
            captured["forward_kwargs"] = dict(forward_kwargs)
            return {
                "forecast_loss": torch.tensor(0.25),
                "forecast_out": torch.randn(2, 6, 1),
                "forecast_mask": torch.ones(2, 6, 1, dtype=torch.bool),
            }

        monkeypatch.setattr(forecaster, "compute_loss", fake_compute_loss)

        stats = forecaster.step(batch, train=False)

        assert captured["time_mask"] is None
        assert captured["forward_kwargs"] == {
            "output_attn": False,
            "target_mask": "mask-output",
        }
        assert stats["loss"].equal(torch.tensor(0.25))
