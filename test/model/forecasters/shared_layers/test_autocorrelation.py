"""Tests for picid.model.forecasters.shared_layers.AutoCorrelation.

``time_delay_agg_inference`` and ``time_delay_agg_full`` call ``.cuda()`` on an
index tensor.  The ``patch_cuda`` fixture monkeypatches ``torch.Tensor.cuda``
to return ``self`` so all tests run on CPU without a GPU.
"""

import pytest
import torch

from picid.model.forecasters.shared_layers.AutoCorrelation import (
    AutoCorrelation,
    AutoCorrelationLayer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, H, C, L = 2, 2, 4, 16  # batch, heads, channels, length


@pytest.fixture(autouse=True)
def patch_cuda(monkeypatch):
    """Make Tensor.cuda() a no-op so tests run on CPU-only machines."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)


def _rand(*shape):
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# time_delay_agg_training
# ---------------------------------------------------------------------------


class TestTimeDelayAggTraining:
    def test_output_shape(self):
        model = AutoCorrelation(factor=1)
        values = _rand(B, H, C, L)
        corr = _rand(B, H, C, L)
        out = model.time_delay_agg_training(values, corr)
        assert out.shape == (B, H, C, L)

    def test_output_is_float(self):
        model = AutoCorrelation(factor=1)
        out = model.time_delay_agg_training(_rand(B, H, C, L), _rand(B, H, C, L))
        assert out.dtype == torch.float32

    def test_factor_2_still_runs(self):
        model = AutoCorrelation(factor=2)
        out = model.time_delay_agg_training(_rand(B, H, C, L), _rand(B, H, C, L))
        assert out.shape == (B, H, C, L)


# ---------------------------------------------------------------------------
# time_delay_agg_inference  (needs patch_cuda)
# ---------------------------------------------------------------------------


class TestTimeDelayAggInference:
    def test_output_shape(self):
        model = AutoCorrelation(factor=1)
        values = _rand(B, H, C, L)
        corr = _rand(B, H, C, L)
        out = model.time_delay_agg_inference(values, corr)
        assert out.shape == (B, H, C, L)

    def test_output_is_float(self):
        model = AutoCorrelation(factor=1)
        out = model.time_delay_agg_inference(_rand(B, H, C, L), _rand(B, H, C, L))
        assert out.dtype == torch.float32

    def test_factor_2(self):
        model = AutoCorrelation(factor=2)
        out = model.time_delay_agg_inference(_rand(B, H, C, L), _rand(B, H, C, L))
        assert out.shape == (B, H, C, L)


# ---------------------------------------------------------------------------
# time_delay_agg_full  (needs patch_cuda)
# ---------------------------------------------------------------------------


class TestTimeDelayAggFull:
    def test_output_shape(self):
        model = AutoCorrelation(factor=1)
        values = _rand(B, H, C, L)
        corr = _rand(B, H, C, L)
        out = model.time_delay_agg_full(values, corr)
        assert out.shape == (B, H, C, L)

    def test_output_is_float(self):
        model = AutoCorrelation(factor=1)
        out = model.time_delay_agg_full(_rand(B, H, C, L), _rand(B, H, C, L))
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# AutoCorrelation.forward — training mode  (no cuda needed)
# ---------------------------------------------------------------------------


def _make_qkv(b, seq, h, e, src=None):
    src = src or seq
    q = _rand(b, seq, h, e)
    k = _rand(b, src, h, e)
    v = _rand(b, src, h, e)
    return q, k, v


class TestAutoCorrelationForwardTrain:
    """Tests run with model.training=True (PyTorch default)."""

    def test_output_shape_l_equals_s(self):
        model = AutoCorrelation(factor=1, output_attention=False)
        q, k, v = _make_qkv(B, L, H, C)
        out, attn = model(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, C)
        assert attn is None

    def test_output_attention_true(self):
        model = AutoCorrelation(factor=1, output_attention=True)
        q, k, v = _make_qkv(B, L, H, C)
        out, attn = model(q, k, v, attn_mask=None)
        assert attn is not None
        assert attn.shape == (B, L, H, C)

    def test_l_greater_than_s_pads_values_and_keys(self):
        """When L > S, zeros are appended to values and keys."""
        tgt, src = L, L // 2
        model = AutoCorrelation(factor=1, output_attention=False)
        q = _rand(B, tgt, H, C)
        k = _rand(B, src, H, C)
        v = _rand(B, src, H, C)
        out, _ = model(q, k, v, attn_mask=None)
        assert out.shape == (B, tgt, H, C)

    def test_l_less_than_s_truncates_values_and_keys(self):
        """When L < S, values and keys are truncated to L."""
        tgt, src = L // 2, L
        model = AutoCorrelation(factor=1, output_attention=False)
        q = _rand(B, tgt, H, C)
        k = _rand(B, src, H, C)
        v = _rand(B, src, H, C)
        out, _ = model(q, k, v, attn_mask=None)
        assert out.shape == (B, tgt, H, C)

    def test_with_scale_set(self):
        model = AutoCorrelation(factor=1, scale=0.5)
        q, k, v = _make_qkv(B, L, H, C)
        out, _ = model(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, C)


# ---------------------------------------------------------------------------
# AutoCorrelation.forward — eval mode  (needs patch_cuda)
# ---------------------------------------------------------------------------


class TestAutoCorrelationForwardEval:
    def test_output_shape_eval_mode(self):
        model = AutoCorrelation(factor=1, output_attention=False).eval()
        q, k, v = _make_qkv(B, L, H, C)
        out, attn = model(q, k, v, attn_mask=None)
        assert out.shape == (B, L, H, C)
        assert attn is None

    def test_output_attention_true_eval_mode(self):
        model = AutoCorrelation(factor=1, output_attention=True).eval()
        q, k, v = _make_qkv(B, L, H, C)
        out, attn = model(q, k, v, attn_mask=None)
        assert attn is not None

    def test_l_greater_s_eval_mode(self):
        tgt, src = L, L // 2
        model = AutoCorrelation(factor=1).eval()
        out, _ = model(
            _rand(B, tgt, H, C), _rand(B, src, H, C), _rand(B, src, H, C), None
        )
        assert out.shape == (B, tgt, H, C)


# ---------------------------------------------------------------------------
# AutoCorrelationLayer
# ---------------------------------------------------------------------------


class TestAutoCorrelationLayer:
    def _layer(self, output_attention=False):
        d_model = H * C
        inner = AutoCorrelation(factor=1, output_attention=output_attention)
        return AutoCorrelationLayer(
            correlation=inner,
            d_model=d_model,
            n_heads=H,
        )

    def test_forward_output_shape(self):
        layer = self._layer()
        x = _rand(B, L, H * C)
        out, attn = layer(x, x, x, attn_mask=None)
        assert out.shape == (B, L, H * C)
        assert attn is None

    def test_forward_output_attention(self):
        layer = self._layer(output_attention=True)
        x = _rand(B, L, H * C)
        out, attn = layer(x, x, x, attn_mask=None)
        assert attn is not None

    def test_custom_d_keys_d_values(self):
        d_model, n_heads, d_k, d_v = 16, 2, 4, 4
        inner = AutoCorrelation(factor=1)
        layer = AutoCorrelationLayer(
            correlation=inner,
            d_model=d_model,
            n_heads=n_heads,
            d_keys=d_k,
            d_values=d_v,
        )
        x = _rand(B, L, d_model)
        out, _ = layer(x, x, x, attn_mask=None)
        assert out.shape == (B, L, d_model)

    def test_layer_init_stores_n_heads(self):
        layer = self._layer()
        assert layer.n_heads == H
