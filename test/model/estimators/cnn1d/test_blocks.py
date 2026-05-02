"""Tests for picid.model.estimators.cnn1d.blocks."""

from __future__ import annotations

import pytest
import torch

from picid.model.estimators.cnn1d.blocks import ResidualBlock


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
