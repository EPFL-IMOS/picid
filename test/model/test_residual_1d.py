"""Tests for picid.model.autoencoder.residual_1d."""

from __future__ import annotations

import pytest
import torch

from picid.model.autoencoder.residual_1d import (
    ResidualBlock,
    ResidualTransposeBlock,
    SymmetricalAutoencoder,
)


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResidualBlock:
    """Tests for ResidualBlock __init__ and forward variants."""

    def _block(self, in_ch=4, out_ch=4, k=3, stride=1, dilation=1, **kwargs):
        return ResidualBlock(
            in_ch, out_ch, k, stride=stride, dilation=dilation, **kwargs
        )

    def _x(self, in_ch=4, length=32):
        return torch.randn(2, in_ch, length)

    def test_default_forward_shape_preserved(self):
        b = self._block()
        out = b(self._x())
        assert out.shape == (2, 4, 32)

    def test_group_norm_type(self):
        b = self._block(norm_type="group")
        assert isinstance(b.norm, torch.nn.GroupNorm)

    def test_batch_norm_type(self):
        b = self._block(norm_type="batch")
        assert isinstance(b.norm, torch.nn.BatchNorm1d)

    def test_none_norm_type_string(self):
        b = self._block(norm_type="none")
        assert isinstance(b.norm, torch.nn.Identity)

    def test_none_norm_type_none_value(self):
        b = self._block(norm_type=None)
        assert isinstance(b.norm, torch.nn.Identity)

    def test_invalid_norm_type_raises(self):
        with pytest.raises(ValueError, match="Unknown norm_type"):
            self._block(norm_type="spectral")

    def test_relu_activation(self):
        b = self._block(activation="relu")
        assert isinstance(b.act, torch.nn.ReLU)

    def test_gelu_activation(self):
        b = self._block(activation="gelu")
        assert isinstance(b.act, torch.nn.GELU)
        out = b(self._x())
        assert out.shape == (2, 4, 32)

    def test_silu_activation(self):
        b = self._block(activation="silu")
        assert isinstance(b.act, torch.nn.SiLU)
        out = b(self._x())
        assert out.shape == (2, 4, 32)

    def test_swish_alias_for_silu(self):
        b = self._block(activation="swish")
        assert isinstance(b.act, torch.nn.SiLU)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            self._block(activation="tanh")

    def test_dropout_enabled_when_prob_positive(self):
        b = self._block(dropout_prob=0.5)
        assert isinstance(b.dropout, torch.nn.Dropout)

    def test_dropout_disabled_when_prob_zero(self):
        b = self._block(dropout_prob=0.0)
        assert isinstance(b.dropout, torch.nn.Identity)

    def test_skip_connection_created_when_channels_differ(self):
        b = self._block(in_ch=4, out_ch=8)
        assert b.skip_connection is not None
        out = b(self._x(in_ch=4))
        assert out.shape == (2, 8, 32)

    def test_skip_connection_created_when_stride_not_one(self):
        b = self._block(in_ch=4, out_ch=4, stride=2)
        assert b.skip_connection is not None
        out = b(self._x(in_ch=4, length=32))
        assert out.shape[0] == 2

    def test_no_skip_connection_when_same_channels_stride_one(self):
        b = self._block(in_ch=4, out_ch=4, stride=1)
        assert b.skip_connection is None

    def test_dilation_forward(self):
        b = self._block(dilation=2)
        out = b(self._x())
        assert out.shape == (2, 4, 32)

    def test_even_kernel_forward(self):
        b = self._block(k=4)
        out = b(self._x())
        assert out.shape == (2, 4, 32)


# ---------------------------------------------------------------------------
# ResidualTransposeBlock
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResidualTransposeBlock:
    """Tests for ResidualTransposeBlock __init__ and forward variants."""

    def _block(self, in_ch=8, out_ch=4, k=3, stride=2, dilation=1, **kwargs):
        return ResidualTransposeBlock(
            in_ch,
            out_ch,
            k,
            stride=stride,
            dilation=dilation,
            dropout_prob=0.0,
            **kwargs,
        )

    def test_basic_forward_upsampling(self):
        b = self._block()
        x = torch.randn(2, 8, 16)
        out = b(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 4

    def test_batch_norm_type(self):
        b = self._block(norm_type="batch")
        assert isinstance(b.norm, torch.nn.BatchNorm1d)

    def test_none_norm_type(self):
        b = self._block(norm_type=None)
        assert isinstance(b.norm, torch.nn.Identity)

    def test_invalid_norm_type_raises(self):
        with pytest.raises(ValueError, match="Unknown norm_type"):
            self._block(norm_type="bad")

    def test_gelu_activation(self):
        b = self._block(activation="gelu")
        assert isinstance(b.act, torch.nn.GELU)
        out = b(torch.randn(2, 8, 16))
        assert out.shape[0] == 2

    def test_silu_activation(self):
        b = self._block(activation="silu")
        assert isinstance(b.act, torch.nn.SiLU)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            self._block(activation="elu")

    def test_dropout_enabled(self):
        b = ResidualTransposeBlock(8, 4, 3, stride=2, dilation=1, dropout_prob=0.3)
        assert isinstance(b.dropout, torch.nn.Dropout)

    def test_skip_connection_when_channels_differ(self):
        b = self._block(in_ch=8, out_ch=4, stride=1)
        assert b.skip_connection is not None

    def test_no_skip_connection_same_channels_stride_one(self):
        b = self._block(in_ch=4, out_ch=4, stride=1)
        assert b.skip_connection is None
        out = b(torch.randn(2, 4, 16))
        assert out.shape[0] == 2

    def test_even_kernel_stride_eq1_trims_to_input_len(self):
        """Even kernel + stride == 1 → output trimmed to input length (line 226-227)."""
        b = self._block(in_ch=8, out_ch=4, k=4, stride=1)
        x = torch.randn(2, 8, 16)
        out = b(x)
        assert out.shape[-1] == x.shape[-1]


# ---------------------------------------------------------------------------
# SymmetricalAutoencoder
# ---------------------------------------------------------------------------


def _minimal_config():
    return {
        "input_channels": 3,
        "input_seq_len": 32,
        "latent_dim": 16,
        "dropout_prob": 0.0,
        "norm_type": "group",
        "activation": "relu",
        "output_channels": [8, 16],
        "kernels": [3, 3],
        "strides": [2, 2],
        "dilations": [1, 1],
    }


@pytest.mark.unit
class TestSymmetricalAutoencoder:
    """Tests for SymmetricalAutoencoder encode, decode, and forward."""

    def test_encode_returns_latent_vector(self):
        model = SymmetricalAutoencoder(_minimal_config())
        x = torch.randn(4, 32, 3)
        latent = model.encode(x)
        assert latent.shape == (4, 16)

    def test_decode_returns_sequence(self):
        model = SymmetricalAutoencoder(_minimal_config())
        z = torch.randn(4, 16)
        out = model.decode(z)
        assert out.shape[0] == 4
        assert out.shape[2] == 3

    def test_forward_shape_preserves_input(self):
        model = SymmetricalAutoencoder(_minimal_config())
        x = torch.randn(4, 32, 3)
        out = model.forward(x)
        assert out.shape == x.shape

    def test_with_gelu_activation(self):
        cfg = {**_minimal_config(), "activation": "gelu"}
        model = SymmetricalAutoencoder(cfg)
        x = torch.randn(2, 32, 3)
        assert model.forward(x).shape == x.shape

    def test_with_batch_norm(self):
        cfg = {**_minimal_config(), "norm_type": "batch"}
        model = SymmetricalAutoencoder(cfg)
        x = torch.randn(2, 32, 3)
        assert model.forward(x).shape == x.shape

    def test_with_dropout(self):
        cfg = {**_minimal_config(), "dropout_prob": 0.1}
        model = SymmetricalAutoencoder(cfg)
        x = torch.randn(2, 32, 3)
        assert model.forward(x).shape == x.shape
