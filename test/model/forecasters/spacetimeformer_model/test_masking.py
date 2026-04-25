"""Tests for spacetimeformer masking utilities (TriangularCausalMask, ProbMask)."""

import torch

from picid.model.forecasters.spacetimeformer_model.utils.masking import (
    ProbMask,
    TriangularCausalMask,
)


class TestTriangularCausalMask:
    def test_init_shape(self):
        mask = TriangularCausalMask(B=2, L=4, device="cpu")
        assert mask.mask.shape == (2, 1, 4, 4)

    def test_mask_upper_triangular(self):
        mask = TriangularCausalMask(B=1, L=3, device="cpu")
        m = mask.mask[0, 0]
        # Upper triangular: diagonal and below are False, above are True
        assert m[0, 0].item() is False
        assert m[0, 1].item() is True
        assert m[0, 2].item() is True
        assert m[1, 0].item() is False
        assert m[1, 1].item() is False
        assert m[1, 2].item() is True
        assert m[2, 0].item() is False
        assert m[2, 1].item() is False
        assert m[2, 2].item() is False

    def test_mask_property(self):
        mask = TriangularCausalMask(B=2, L=5, device="cpu")
        assert mask.mask is not None
        assert mask.mask.dtype == torch.bool


class TestProbMask:
    def test_init_shape(self):
        """ProbMask: scores (B, H, n_top, L_K), index (B, H, n_top)."""
        B, H, L, n_top = 2, 2, 4, 2
        scores = torch.randn(B, H, n_top, L)
        index = torch.randint(0, L, (B, H, n_top))
        pm = ProbMask(B=B, H=H, L=L, index=index, scores=scores, device="cpu")
        assert pm.mask.shape == scores.shape

    def test_mask_property(self):
        B, H, L, n_top = 1, 1, 3, 2
        scores = torch.randn(B, H, n_top, L)
        index = torch.zeros((B, H, n_top), dtype=torch.long)
        pm = ProbMask(B=B, H=H, L=L, index=index, scores=scores, device="cpu")
        assert pm.mask is not None
        assert pm.mask.dtype == torch.bool
