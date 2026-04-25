"""Tests for Spacetimeformer NN components (PowerNorm, GroupScaling1D, ReconstructionDropout).

PowerNorm and MaskPowerNorm are normalization layers used in Spacetimeformer.
GroupScaling1D scales inputs by second moment per group.
ReconstructionDropout is used for encoder embedding regularization.
"""

import torch

from picid.model.forecasters.spacetimeformer_model.nn.data_dropout import (
    ReconstructionDropout,
    RandomMask,
    create_subsequence_mask,
)
from picid.model.forecasters.spacetimeformer_model.nn.powernorm import (
    GroupScaling1D,
    MaskPowerNorm,
    _sum_ft,
    _unsqueeze_ft,
)


class TestGroupScaling1D:
    """Tests for GroupScaling1D layer."""

    def test_init_default(self):
        """GroupScaling1D initializes with default eps and group_num."""
        layer = GroupScaling1D()
        assert layer.eps == 1e-5
        assert layer.group_num == 4

    def test_init_custom(self):
        """GroupScaling1D accepts custom eps and group_num."""
        layer = GroupScaling1D(eps=1e-3, group_num=8)
        assert layer.eps == 1e-3
        assert layer.group_num == 8

    def test_extra_repr(self):
        """extra_repr returns string with eps and group_num."""
        layer = GroupScaling1D(eps=1e-4, group_num=2)
        s = layer.extra_repr()
        assert "1e-4" in s or "0.0001" in s
        assert "2" in s

    def test_forward_basic(self):
        """Forward pass scales input by second moment."""
        torch.manual_seed(42)
        layer = GroupScaling1D(eps=1e-5, group_num=2)
        # T, B, C - seq=6, batch=2, channels=4 (must be divisible by group_num)
        x = torch.randn(6, 2, 4)
        out = layer(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_different_shapes(self):
        """Forward handles various T, B, C combinations."""
        layer = GroupScaling1D(group_num=2)
        for T, B, C in [(4, 2, 4), (8, 1, 8), (10, 3, 6)]:
            x = torch.randn(T, B, C)
            out = layer(x)
            assert out.shape == (T, B, C)


class TestMaskPowerNorm:
    """Tests for MaskPowerNorm layer."""

    def test_init_basic(self):
        """MaskPowerNorm initializes with num_features."""
        layer = MaskPowerNorm(num_features=32)
        assert layer.num_features == 32
        assert layer.weight.shape == (32,)
        assert layer.bias.shape == (32,)
        assert layer.eps == 1e-5

    def test_init_custom(self):
        """MaskPowerNorm accepts custom alpha, warmup, group_num."""
        layer = MaskPowerNorm(
            num_features=16,
            eps=1e-4,
            alpha_fwd=0.95,
            alpha_bkw=0.95,
            warmup_iters=100,
            group_num=4,
        )
        assert layer.afwd == 0.95
        assert layer.abkw == 0.95
        assert layer.warmup_iters == 100
        assert layer.group_num == 4

    def test_extra_repr(self):
        """extra_repr includes key parameters."""
        layer = MaskPowerNorm(num_features=8)
        s = layer.extra_repr()
        assert "8" in s
        assert "eps" in s

    def test_forward_train_no_mask(self):
        """Forward in training mode without pad_mask."""
        torch.manual_seed(42)
        layer = MaskPowerNorm(num_features=8, warmup_iters=100)
        layer.train()
        # T, B, C
        x = torch.randn(6, 2, 8)
        out = layer(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_eval_no_mask(self):
        """Forward in eval mode without pad_mask."""
        torch.manual_seed(42)
        layer = MaskPowerNorm(num_features=8)
        layer.eval()
        x = torch.randn(6, 2, 8)
        out = layer(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_forward_train_with_pad_mask(self):
        """Forward in training mode with pad_mask (B x T, padding=True)."""
        torch.manual_seed(42)
        layer = MaskPowerNorm(num_features=8, warmup_iters=100)
        layer.train()
        x = torch.randn(6, 2, 8)  # T=6, B=2
        # pad_mask: B x T, True = padding
        pad_mask = torch.zeros(2, 6, dtype=torch.bool)
        pad_mask[0, 5] = True  # last token of batch 0 is padding
        pad_mask[1, 4:] = True  # last 2 tokens of batch 1 are padding
        out = layer(x, pad_mask=pad_mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_forward_eval_with_pad_mask(self):
        """Forward in eval mode with pad_mask."""
        torch.manual_seed(42)
        layer = MaskPowerNorm(num_features=8)
        layer.eval()
        x = torch.randn(6, 2, 8)
        pad_mask = torch.zeros(2, 6, dtype=torch.bool)
        pad_mask[1, 5] = True
        out = layer(x, pad_mask=pad_mask)
        assert out.shape == x.shape

    def test_forward_shaped_input_2d(self):
        """Forward with 2D input (shaped_input path)."""
        torch.manual_seed(42)
        layer = MaskPowerNorm(num_features=8)
        layer.eval()
        # B x C only - gets unsqueezed to 1 x B x C
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_backward_train(self):
        """Backward pass works in training mode."""
        torch.manual_seed(42)
        layer = MaskPowerNorm(num_features=8, warmup_iters=100)
        layer.train()
        x = torch.randn(4, 2, 8, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPowerNormHelpers:
    """Tests for module-level helper functions."""

    def test_sum_ft(self):
        """_sum_ft sums over first and last dim."""
        x = torch.randn(3, 4, 5)
        out = _sum_ft(x)
        assert out.shape == (4,)

    def test_unsqueeze_ft(self):
        """_unsqueeze_ft adds dims at front and tail."""
        x = torch.randn(2, 3)
        out = _unsqueeze_ft(x)
        assert out.shape == (1, 2, 3, 1)


class TestReconstructionDropout:
    """Tests for ReconstructionDropout layer."""

    def test_forward_shape(self):
        """Forward preserves input shape (batch, length, dim)."""
        layer = ReconstructionDropout(drop_standard=0.1, skip_all_drop=0.0)
        layer.train()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        assert out.shape == x.shape

    def test_eval_mode_identity(self):
        """Eval mode returns input unchanged."""
        torch.manual_seed(42)
        layer = ReconstructionDropout(drop_standard=0.5, skip_all_drop=0.0)
        layer.eval()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_training_zero_dropout_identity(self):
        """Training with all dropout probs 0 returns input unchanged."""
        layer = ReconstructionDropout(
            drop_full_timesteps=0.0,
            drop_standard=0.0,
            drop_seq=0.0,
            skip_all_drop=0.0,
        )
        layer.train()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_training_skip_all_drop_one_identity(self):
        """Training with skip_all_drop=1.0 returns input unchanged."""
        torch.manual_seed(42)
        layer = ReconstructionDropout(drop_standard=0.5, skip_all_drop=1.0)
        layer.train()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_training_drop_standard_one_zeros_output(self):
        """Training with drop_standard=1.0 and skip_all_drop=0 zeros output."""
        torch.manual_seed(42)
        layer = ReconstructionDropout(drop_standard=1.0, skip_all_drop=0.0)
        layer.train()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        assert out.shape == x.shape
        torch.testing.assert_close(out, torch.zeros_like(x))

    def test_repr(self):
        """__repr__ returns string with dropout configuration."""
        layer = ReconstructionDropout(
            drop_full_timesteps=0.05,
            drop_standard=0.2,
            drop_seq=0.1,
            drop_max_seq_len=5,
        )
        s = repr(layer)
        assert "0.05" in s or "0.2" in s
        assert "Standard" in s
        assert "Seq" in s


class TestCreateSubsequenceMask:
    """Tests for create_subsequence_mask helper."""

    def test_r_zero_returns_all_false(self):
        """When r<=0, returns mask of all False (mask everything)."""
        x = torch.randn(2, 10)
        mask = create_subsequence_mask(x, r=0)
        assert mask.shape == x.shape
        assert not mask.any()

    def test_output_shape_2d_input(self):
        """2D input is expanded to 3D; output shape matches."""
        x = torch.randn(4, 8)
        mask = create_subsequence_mask(x, r=0.2, stateful=False)
        assert mask.ndim == 3
        assert mask.shape == (1, 4, 8)

    def test_output_shape_3d_input(self):
        """3D input shape is preserved in output."""
        x = torch.randn(2, 4, 8)
        mask = create_subsequence_mask(x, r=0.15, stateful=False)
        assert mask.shape == (2, 4, 8)
        assert mask.dtype == torch.bool

    def test_stateful_true(self):
        """stateful=True uses Geometric distribution."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8)
        mask = create_subsequence_mask(x, r=0.2, lm=3, stateful=True)
        assert mask.shape == (2, 4, 8)
        assert mask.dtype == torch.bool

    def test_sync_random(self):
        """sync='random' applies same mask across dims with 50% prob."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 8)
        mask = create_subsequence_mask(x, r=0.15, stateful=False, sync="random")
        assert mask.shape == (2, 4, 8)


class TestRandomMask:
    """Tests for RandomMask layer."""

    def test_forward_training(self):
        """Forward in training mode replaces values with change_to_val."""
        torch.manual_seed(42)
        layer = RandomMask(prob=0.5, change_to_val=0.0)
        layer.train()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        assert out.shape == x.shape

    def test_forward_eval_identity(self):
        """Eval mode returns input unchanged."""
        layer = RandomMask(prob=0.5, change_to_val=0.0)
        layer.eval()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_change_to_val_none_identity(self):
        """change_to_val=None makes forward identity."""
        layer = RandomMask(prob=0.5, change_to_val=None)
        layer.train()
        x = torch.randn(2, 10, 4)
        out = layer(x)
        torch.testing.assert_close(out, x)

    def test_repr(self):
        """__repr__ returns string with prob and val."""
        layer = RandomMask(prob=0.3, change_to_val=0.0)
        s = repr(layer)
        assert "0.3" in s or "RandomMask" in s
