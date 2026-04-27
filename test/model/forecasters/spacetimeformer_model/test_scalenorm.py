"""Tests for ScaleNorm layer."""

import torch

from picid.model.forecasters.spacetimeformer_model.nn.scalenorm import ScaleNorm


class TestScaleNorm:
    def test_init(self):
        layer = ScaleNorm(dim=64, eps=1e-5)
        assert layer.scale == 64**-0.5
        assert layer.eps == 1e-5
        assert layer.g.shape == (1,)

    def test_forward_shape(self):
        layer = ScaleNorm(dim=32)
        x = torch.randn(2, 10, 32)
        out = layer(x)
        assert out.shape == x.shape

    def test_forward_no_nan(self):
        layer = ScaleNorm(dim=16, eps=1e-5)
        x = torch.randn(3, 5, 16)
        out = layer(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
