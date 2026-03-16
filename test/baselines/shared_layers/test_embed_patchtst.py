"""Tests for Embed_PatchTST (PatchEmbedding, PositionalEmbedding)."""

import torch
from picid.baselines.shared_layers.Embed_PatchTST import PatchEmbedding, PositionalEmbedding


def test_positional_embedding():
    pe = PositionalEmbedding(d_model=16, max_len=100)
    x = torch.randn(2, 50, 16)
    out = pe(x)
    assert out.shape == (1, 50, 16)


def test_patch_embedding():
    pe = PatchEmbedding(d_model=16, patch_len=8, stride=4, padding=4, dropout=0)
    x = torch.randn(2, 4, 96)  # B, nvars, seq
    out, n_vars = pe(x)
    assert n_vars == 4
    assert out.dim() == 3
