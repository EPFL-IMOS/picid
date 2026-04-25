"""Tests for spacetimeformer attention layers (FullAttention, ProbAttention, AttentionLayer)."""

import torch

from picid.model.forecasters.spacetimeformer_model.nn.attn import (
    AttentionLayer,
    FullAttention,
    ProbAttention,
)


class TestFullAttention:
    def test_forward_no_mask(self):
        attn = FullAttention(mask_flag=False)
        B, L, H, E = 2, 8, 2, 16
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        out, attn_weights = attn(q, k, v, attn_mask=None, output_attn=False)
        assert out.shape == (B, L, H, E)
        assert attn_weights is None

    def test_forward_with_mask(self):
        """Pass tensor mask; attn does attn_mask.unsqueeze(1) so pass (B, L, L)."""
        attn = FullAttention(mask_flag=True)
        B, L, H, E = 2, 6, 2, 16
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        mask = torch.triu(torch.ones(B, L, L, dtype=torch.bool), diagonal=1)
        out, attn_weights = attn(q, k, v, attn_mask=mask, output_attn=False)
        assert out.shape == (B, L, H, E)
        assert attn_weights is None

    def test_forward_output_attn(self):
        attn = FullAttention(mask_flag=False)
        B, L, H, E = 2, 4, 2, 8
        q = torch.randn(B, L, H, E)
        k = torch.randn(B, L, H, E)
        v = torch.randn(B, L, H, E)
        out, attn_weights = attn(q, k, v, attn_mask=None, output_attn=True)
        assert out.shape == (B, L, H, E)
        assert attn_weights is not None
        assert attn_weights.shape == (B, H, L, L)


class TestProbAttention:
    def test_forward_mask_flag_false(self):
        attn = ProbAttention(mask_flag=False, factor=2)
        B, L, H, D = 2, 8, 2, 16
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        out, attn_out = attn(q, k, v, attn_mask=None, output_attn=False)
        assert out.shape == (B, L, H, D)
        assert attn_out is None

    def test_forward_mask_flag_true(self):
        attn = ProbAttention(mask_flag=True, factor=2)
        B, L, H, D = 2, 8, 2, 16
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        out, attn_out = attn(q, k, v, attn_mask=None, output_attn=False)
        assert out.shape == (B, L, H, D)

    def test_forward_output_attn(self):
        attn = ProbAttention(mask_flag=False, factor=2)
        B, L, H, D = 2, 8, 2, 16
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        out, attn_out = attn(q, k, v, attn_mask=None, output_attn=True)
        assert out.shape == (B, L, H, D)
        assert attn_out is not None


class TestAttentionLayer:
    def test_forward_basic(self):
        inner = FullAttention
        layer = AttentionLayer(
            attention=inner,
            d_model=32,
            d_queries_keys=16,
            d_values=16,
            n_heads=2,
        )
        B, L, S = 2, 8, 8
        q = torch.randn(B, L, 32)
        k = torch.randn(B, S, 32)
        v = torch.randn(B, S, 32)
        out, attn = layer(q, k, v, attn_mask=None, output_attn=False)
        assert out.shape == (B, L, 32)
        assert attn is None

    def test_forward_output_attn(self):
        inner = FullAttention
        layer = AttentionLayer(
            attention=inner,
            d_model=32,
            d_queries_keys=16,
            d_values=16,
            n_heads=2,
        )
        B, L, S = 2, 6, 6
        q = torch.randn(B, L, 32)
        k = torch.randn(B, S, 32)
        v = torch.randn(B, S, 32)
        out, attn = layer(q, k, v, attn_mask=None, output_attn=True)
        assert out.shape == (B, L, 32)
        assert attn is not None
