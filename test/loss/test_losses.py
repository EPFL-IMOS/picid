import pytest
import torch
from lmetk.loss.default import MSELoss
from lmetk.loss.cross_entropy import (
    CrossEntropyLoss,
)

# =========================================================================
# 1. MSE Loss Tests
# =========================================================================


def test_mse_loss_correctness():
    """
    Verifies simple MSE calculation: (y_pred - y_true)^2.
    """
    loss_fn = MSELoss(reduction="mean")

    # Batch=1, Seq=2, Dim=1
    predictions = torch.tensor([[[1.0], [2.0]]])
    targets = torch.tensor([[[2.0], [4.0]]])

    model_out = {"predictions": predictions, "targets": targets}
    batch = {}  # Not used by MSE forward, but required by signature

    result = loss_fn(model_out, batch)

    # Manual Calc:
    # T0: (1.0 - 2.0)^2 = 1.0
    # T1: (2.0 - 4.0)^2 = 4.0
    # Mean: (1.0 + 4.0) / 2 = 2.5
    assert torch.isclose(result["loss"], torch.tensor(2.5))


def test_mse_loss_reduction_sum():
    """Verifies reduction='sum'."""
    loss_fn = MSELoss(reduction="sum")
    predictions = torch.tensor([[[1.0], [2.0]]])
    targets = torch.tensor([[[2.0], [4.0]]])

    model_out = {"predictions": predictions, "targets": targets}
    result = loss_fn(model_out, {})

    # Sum: 1.0 + 4.0 = 5.0
    assert torch.isclose(result["loss"], torch.tensor(5.0))


def test_mse_loss_reduction_none():
    """Verifies reduction='none' returns element-wise loss."""
    loss_fn = MSELoss(reduction="none")
    predictions = torch.tensor([[[1.0], [2.0]]])
    targets = torch.tensor([[[2.0], [4.0]]])

    model_out = {"predictions": predictions, "targets": targets}
    result = loss_fn(model_out, {})

    expected = torch.tensor([[[1.0], [4.0]]])
    assert torch.allclose(result["loss"], expected)


# =========================================================================
# 2. Cross Entropy Loss Tests
# =========================================================================


def test_ce_loss_shape_handling_3d_targets():
    """
    Verifies handling of 3D targets (Batch, Seq, 1).
    The implementation should squeeze the last dimension.
    """
    loss_fn = CrossEntropyLoss()

    # Batch=1, Seq=2, Classes=3
    # Logits favour Class 2 at T0 and Class 0 at T1
    predictions = torch.tensor([[[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]]])

    # Targets: (Batch, Seq, 1) -> Class 2, Class 0
    targets = torch.tensor([[[2], [0]]])

    model_out = {"predictions": predictions, "targets": targets}
    result = loss_fn(model_out, {})

    # Loss should be very small (near 0) because predictions match targets perfectly
    assert result["loss"].item() < 0.01


def test_ce_loss_shape_handling_2d_targets():
    """
    Verifies handling of 2D targets (Batch, Seq).
    Standard PyTorch format.
    """
    loss_fn = CrossEntropyLoss()

    predictions = torch.randn(2, 5, 4)  # (B=2, S=5, C=4)
    targets = torch.randint(0, 4, (2, 5))  # (B=2, S=5)

    model_out = {"predictions": predictions, "targets": targets}

    # Should run without error
    result = loss_fn(model_out, {})
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].dim() == 0  # Scalar mean


def test_ce_loss_permutation_logic():
    """
    Verifies that the loss permutes (B, S, C) to (B, C, S) internally.
    We test this by providing a known mismatch if permuted incorrectly.
    """
    loss_fn = CrossEntropyLoss()

    # Batch=1, Seq=1, Classes=3
    # Pred: [100, 0, 0] -> Strong Class 0
    predictions = torch.tensor([[[100.0, 0.0, 0.0]]])

    # Target: Class 0
    targets = torch.tensor([[0]])

    model_out = {"predictions": predictions, "targets": targets}
    result = loss_fn(model_out, {})

    # If permuted correctly:
    #   Pred becomes (1, 3, 1) -> Class dim is index 1.
    #   CrossEntropy sees Class 0 has score 100. Loss ~ 0.
    # If NOT permuted:
    #   It might treat Sequence dimension as class dimension (if shapes allowed)
    #   or crash. Since dim=1 here, it's ambiguous, so let's check value.
    assert result["loss"].item() < 0.001


def test_ce_loss_ignore_index():
    """
    Verifies the ignore_index functionality (masking).
    """
    loss_fn = CrossEntropyLoss(ignore_index=-100)

    # Seq Length 2
    # T0: Wrong prediction (should yield high loss)
    # T1: Wrong prediction BUT target is -100 (should be ignored)
    predictions = torch.tensor([[[0.0, 10.0], [0.0, 10.0]]])  # Predicts Class 1 always
    targets = torch.tensor([[0, -100]])  # Target Class 0, then Ignore

    model_out = {"predictions": predictions, "targets": targets}
    result = loss_fn(model_out, {})

    # Calculate expected loss manually for T0 only
    # Softmax([0, 10]) -> [4.5e-5, 0.9999]
    # NLL(Class 0) -> -log(4.5e-5) ~= 10.0
    expected_loss = 10.0

    assert torch.isclose(result["loss"], torch.tensor(expected_loss), atol=0.1)


def test_ce_loss_dtype_casting():
    """
    Verifies that float targets are cast to long/int automatically.
    """
    loss_fn = CrossEntropyLoss()

    predictions = torch.randn(1, 2, 3)
    # Targets as Floats (e.g. from a regression dataloader reused for classification)
    targets = torch.tensor([[0.0, 2.0]], dtype=torch.float32)

    model_out = {"predictions": predictions, "targets": targets}

    # Should not raise runtime error about expecting Long
    result = loss_fn(model_out, {})
    assert result["loss"] is not None
