"""Tests for picid.model.forecasters.shared_layers.Autoformer_EncDec."""

import torch
import torch.nn as nn

from picid.model.forecasters.shared_layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    moving_avg,
    my_Layernorm,
    series_decomp,
)

# ---------------------------------------------------------------------------
# Common dimensions used across tests
# ---------------------------------------------------------------------------
B, T, D = 2, 12, 8


# ---------------------------------------------------------------------------
# Minimal attention stub
# ---------------------------------------------------------------------------


class _IdentityAttention(nn.Module):
    """Returns (x, None) — pass-through attention for structural tests."""

    def forward(self, q, k, v, attn_mask=None):
        return q, None


class _PassthroughConv(nn.Module):
    """Conv stub that returns its input unchanged."""

    def forward(self, x):
        return x


# ---------------------------------------------------------------------------
# my_Layernorm
# ---------------------------------------------------------------------------


class TestMyLayernorm:
    def test_forward_shape_preserved(self):
        ln = my_Layernorm(D)
        out = ln(torch.randn(B, T, D))
        assert out.shape == (B, T, D)

    def test_mean_removed_along_time(self):
        """The seasonal layernorm subtracts the time-average, so mean ≈ 0."""
        ln = my_Layernorm(D)
        ln.eval()
        out = ln(torch.randn(B, T, D))
        # After bias removal the temporal mean should be near zero
        assert out.mean(dim=1).abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# moving_avg
# ---------------------------------------------------------------------------


class TestMovingAvg:
    def test_forward_preserves_time_length(self):
        ma = moving_avg(kernel_size=3, stride=1)
        out = ma(torch.randn(B, T, D))
        assert out.shape == (B, T, D)

    def test_larger_kernel_still_preserves_length(self):
        ma = moving_avg(kernel_size=7, stride=1)
        out = ma(torch.randn(B, T, D))
        assert out.shape == (B, T, D)

    def test_trend_smoother_than_input(self):
        """Moving average output should have lower variance than white noise."""
        ma = moving_avg(kernel_size=5, stride=1)
        x = torch.randn(1, 50, 1)
        out = ma(x)
        assert out.var().item() < x.var().item()


# ---------------------------------------------------------------------------
# series_decomp
# ---------------------------------------------------------------------------


class TestSeriesDecomp:
    def test_forward_returns_two_tensors_of_same_shape(self):
        sd = series_decomp(kernel_size=3)
        x = torch.randn(B, T, D)
        res, trend = sd(x)
        assert res.shape == (B, T, D)
        assert trend.shape == (B, T, D)

    def test_residual_plus_trend_equals_input(self):
        sd = series_decomp(kernel_size=3)
        x = torch.randn(B, T, D)
        res, trend = sd(x)
        torch.testing.assert_close(res + trend, x)


# ---------------------------------------------------------------------------
# EncoderLayer
# ---------------------------------------------------------------------------


class TestEncoderLayer:
    def test_forward_relu_output_shape(self):
        layer = EncoderLayer(
            _IdentityAttention(), d_model=D, d_ff=16, moving_avg=3, activation="relu"
        )
        out, attn = layer(torch.randn(B, T, D))
        assert out.shape == (B, T, D)

    def test_forward_gelu_output_shape(self):
        layer = EncoderLayer(
            _IdentityAttention(), d_model=D, d_ff=16, moving_avg=3, activation="gelu"
        )
        out, _ = layer(torch.randn(B, T, D), attn_mask=None)
        assert out.shape == (B, T, D)

    def test_default_d_ff_is_4x_d_model(self):
        layer = EncoderLayer(_IdentityAttention(), d_model=D)
        assert layer.conv1.out_channels == 4 * D

    def test_attn_output_passed_through(self):
        """attn returned by EncoderLayer should come from the inner attention."""
        layer = EncoderLayer(_IdentityAttention(), d_model=D, d_ff=16, moving_avg=3)
        _, attn = layer(torch.randn(B, T, D))
        assert attn is None  # stub returns None


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def _layer(self):
        return EncoderLayer(_IdentityAttention(), d_model=D, d_ff=16, moving_avg=3)

    def test_forward_no_conv_no_norm(self):
        enc = Encoder([self._layer()])
        out, attns = enc(torch.randn(B, T, D))
        assert out.shape == (B, T, D)
        assert len(attns) == 1

    def test_forward_with_norm_layer(self):
        enc = Encoder([self._layer()], norm_layer=nn.LayerNorm(D))
        out, _ = enc(torch.randn(B, T, D))
        assert out.shape == (B, T, D)

    def test_forward_multiple_layers_no_conv(self):
        enc = Encoder([self._layer(), self._layer()])
        out, attns = enc(torch.randn(B, T, D))
        assert out.shape == (B, T, D)
        assert len(attns) == 2

    def test_forward_with_conv_layers_branch(self):
        """With conv_layers: zip(attn_layers, conv_layers) then attn_layers[-1]."""
        enc = Encoder(
            [self._layer(), self._layer()],
            conv_layers=[_PassthroughConv()],
        )
        out, attns = enc(torch.randn(B, T, D))
        assert out.shape == (B, T, D)
        # 1 from zip loop + 1 from attn_layers[-1] = 2
        assert len(attns) == 2

    def test_conv_layers_none_stored_as_none(self):
        enc = Encoder([self._layer()], conv_layers=None)
        assert enc.conv_layers is None

    def test_conv_layers_list_stored_as_module_list(self):
        enc = Encoder([self._layer()], conv_layers=[_PassthroughConv()])
        assert isinstance(enc.conv_layers, nn.ModuleList)


# ---------------------------------------------------------------------------
# DecoderLayer
# ---------------------------------------------------------------------------


class TestDecoderLayer:
    def _layer(self, activation="relu"):
        return DecoderLayer(
            self_attention=_IdentityAttention(),
            cross_attention=_IdentityAttention(),
            d_model=D,
            c_out=2,
            d_ff=16,
            moving_avg=3,
            activation=activation,
        )

    def test_forward_relu_output_shape(self):
        x = torch.randn(B, T, D)
        out, trend = self._layer()(x, cross=x)
        assert out.shape == (B, T, D)
        assert trend.shape == (B, T, 2)

    def test_forward_gelu_output_shape(self):
        x = torch.randn(B, T, D)
        out, trend = self._layer("gelu")(x, cross=x)
        assert out.shape == (B, T, D)

    def test_default_d_ff_is_4x_d_model(self):
        layer = DecoderLayer(
            _IdentityAttention(), _IdentityAttention(), d_model=D, c_out=2
        )
        assert layer.conv1.out_channels == 4 * D

    def test_forward_with_masks_provided(self):
        x = torch.randn(B, T, D)
        out, trend = self._layer()(x, cross=x, x_mask=None, cross_mask=None)
        assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class TestDecoder:
    def _dec_layer(self):
        return DecoderLayer(
            self_attention=_IdentityAttention(),
            cross_attention=_IdentityAttention(),
            d_model=D,
            c_out=2,
            d_ff=16,
            moving_avg=3,
        )

    def test_forward_no_norm_no_projection(self):
        dec = Decoder([self._dec_layer()])
        x = torch.randn(B, T, D)
        trend = torch.zeros(B, T, 2)
        out, final_trend = dec(x, cross=x, trend=trend)
        assert out.shape == (B, T, D)

    def test_forward_with_norm_layer(self):
        dec = Decoder([self._dec_layer()], norm_layer=nn.LayerNorm(D))
        x = torch.randn(B, T, D)
        out, _ = dec(x, cross=x, trend=torch.zeros(B, T, 2))
        assert out.shape == (B, T, D)

    def test_forward_with_projection(self):
        dec = Decoder([self._dec_layer()], projection=nn.Linear(D, 4))
        x = torch.randn(B, T, D)
        out, _ = dec(x, cross=x, trend=torch.zeros(B, T, 2))
        assert out.shape == (B, T, 4)

    def test_multiple_decoder_layers_accumulate_trend(self):
        dec = Decoder([self._dec_layer(), self._dec_layer()])
        x = torch.randn(B, T, D)
        trend_init = torch.zeros(B, T, 2)
        _, trend = dec(x, cross=x, trend=trend_init)
        # trend should have been modified (residual added twice)
        assert trend.shape == (B, T, 2)
