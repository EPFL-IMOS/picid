"""Tests for RevIN, MovingAvg, SeriesDecomposition (picid.model.utils.revin)."""

import pytest
import torch

from picid.model.utils.revin import RevIN, MovingAvg, SeriesDecomposition


class TestMovingAvg:
    def test_forward_shape(self):
        m = MovingAvg(kernel_size=5, stride=1)
        x = torch.randn(2, 10, 3)
        out = m(x)
        assert out.shape == x.shape

    def test_forward_reduces_variance(self):
        m = MovingAvg(kernel_size=5, stride=1)
        x = torch.randn(2, 20, 4)
        out = m(x)
        assert out.var() < x.var()


class TestSeriesDecomposition:
    def test_forward_returns_two_tensors(self):
        sd = SeriesDecomposition(kernel_size=5)
        x = torch.randn(2, 10, 3)
        res, trend = sd(x)
        assert res.shape == x.shape
        assert trend.shape == x.shape
        torch.testing.assert_close(res + trend, x, atol=1e-5, rtol=1e-5)


class TestRevIN:
    def test_norm_denorm_roundtrip(self):
        r = RevIN(num_features=4, affine=False)
        x = torch.randn(2, 10, 4)
        normed = r(x, mode="norm", update_stats=True)
        denormed = r(normed, mode="denorm")
        torch.testing.assert_close(denormed, x, atol=1e-5, rtol=1e-5)

    def test_norm_denorm_with_affine(self):
        r = RevIN(num_features=4, affine=True)
        x = torch.randn(2, 10, 4)
        normed = r(x, mode="norm", update_stats=True)
        denormed = r(normed, mode="denorm")
        torch.testing.assert_close(denormed, x, atol=1e-4, rtol=1e-4)

    def test_update_stats_false_reuses_stats(self):
        r = RevIN(num_features=4, affine=False)
        x1 = torch.randn(2, 10, 4) * 10
        r(x1, mode="norm", update_stats=True)
        x2 = torch.randn(2, 5, 4) * 100
        out = r(x2, mode="norm", update_stats=False)
        assert not torch.isnan(out).any()

    def test_raises_bad_mode(self):
        r = RevIN(num_features=4)
        with pytest.raises(NotImplementedError):
            r(torch.randn(2, 10, 4), mode="invalid")

    def test_raises_ndim_not_3(self):
        r = RevIN(num_features=4)
        with pytest.raises(AssertionError):
            r(torch.randn(2, 4), mode="norm", update_stats=True)
