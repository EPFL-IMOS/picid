"""Tests for spacetimeformer extra_layers (Flatten, ConvBlock, Normalization, etc.)."""

import torch

from picid.model.forecasters.spacetimeformer_model.nn.extra_layers import (
    Flatten,
    ConvBlock,
    FoldForPred,
    Localize,
    MakeCrossMaskFromSeq,
    MakeSelfMaskFromSeq,
    Normalization,
    ReverseLocalize,
    ReverseWindowTime,
    Stack,
    WindowTime,
)


class TestFlatten:
    def test_flatten_shape(self):
        x = torch.randn(2, 10, 4)
        out = Flatten(x)
        assert out.shape == (2, 40, 1)


class TestLocalize:
    def test_localize_reverse(self):
        x = torch.randn(2, 10, 4)
        loc = Localize(x, variables=2)
        rev = ReverseLocalize(loc, variables=2)
        assert rev.shape == (2, 10, 4)


class TestMakeSelfMaskFromSeq:
    def test_none_returns_none(self):
        assert MakeSelfMaskFromSeq(None) is None

    def test_mask_shape(self):
        seq_mask = torch.zeros(2, 6, 1)  # B, L, 1
        seq_mask[0, 2] = 1
        mask = MakeSelfMaskFromSeq(seq_mask)
        assert mask.shape == (2, 6, 6)


class TestMakeCrossMaskFromSeq:
    def test_self_none_returns_none(self):
        assert MakeCrossMaskFromSeq(None, torch.randn(2, 4, 1)) is None

    def test_mask_shape(self):
        self_mask = torch.zeros(2, 6, 1)
        cross_mask = torch.zeros(2, 4, 1)
        mask = MakeCrossMaskFromSeq(self_mask, cross_mask)
        assert mask.shape == (2, 6, 4)


class TestWindowTime:
    def test_windows_one_returns_input(self):
        x = torch.randn(2, 24, 4)
        out = WindowTime(x, dy=8, windows=1, window_offset=0)
        assert out is x

    def test_windows_one_none_returns_none(self):
        assert WindowTime(None, dy=8, windows=1, window_offset=0) is None

    def test_windows_gt_one(self):
        """WindowTime: batch (dy len) dim; dy=8, len divisible by windows."""
        x = torch.randn(2, 48, 4)
        out = WindowTime(x, dy=8, windows=2, window_offset=0)
        assert out.shape[0] == 4
        assert out.shape[2] == 4


class TestReverseWindowTime:
    def test_windows_one_returns_input(self):
        x = torch.randn(2, 24, 4)
        assert ReverseWindowTime(x, dy=8, windows=1, window_offset=0) is x

    def test_reverse_roundtrip(self):
        x = torch.randn(2, 48, 4)
        w = WindowTime(x, dy=8, windows=2, window_offset=0)
        rev = ReverseWindowTime(w, dy=8, windows=2, window_offset=0)
        assert rev.shape == x.shape


class TestConvBlock:
    def test_forward_gelu(self):
        """ConvBlock with pool halves length."""
        block = ConvBlock(split_length_into=4, d_model=16, activation="gelu")
        x = torch.randn(2, 40, 16)
        out = block(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 16

    def test_forward_elu(self):
        block = ConvBlock(split_length_into=4, d_model=16, activation="elu")
        x = torch.randn(2, 40, 16)
        out = block(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 16

    def test_forward_relu(self):
        block = ConvBlock(split_length_into=4, d_model=16, activation="relu")
        x = torch.randn(2, 40, 16)
        out = block(x)
        assert out.shape[0] == 2
        assert out.shape[2] == 16


class TestNormalization:
    def test_layer(self):
        norm = Normalization("layer", d_model=16)
        x = torch.randn(2, 10, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_scale(self):
        norm = Normalization("scale", d_model=16)
        x = torch.randn(2, 10, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_batch(self):
        norm = Normalization("batch", d_model=16)
        x = torch.randn(2, 10, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_power(self):
        norm = Normalization("power", d_model=16)
        x = torch.randn(2, 10, 16)
        out = norm(x)
        assert out.shape == x.shape

    def test_none(self):
        norm = Normalization("none")
        x = torch.randn(2, 10, 16)
        out = norm(x)
        assert out.shape == x.shape


class TestStack:
    def test_stack_shape(self):
        x = torch.randn(2, 24, 4)
        out = Stack(x, dy=4)
        assert out.shape == (2, 6, 4, 4)


class TestFoldForPred:
    def test_fold_shape(self):
        """FoldForPred: batch (dy len) dim -> dim batch len dy, squeeze(0) when dim=1."""
        x = torch.randn(2, 24, 1)
        out = FoldForPred(x, dy=4)
        assert out.shape == (2, 6, 4)
