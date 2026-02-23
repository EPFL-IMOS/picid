import pytest
import torch
from picid.loss.default import (
    MSELoss,
    MAELoss,
    HuberLoss,
    QuantileLoss,
    MAPELoss,
    SMAPELoss,
    WeightedMSELoss,
    CombinedLoss,
)
from picid.loss.cross_entropy import (
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


# =========================================================================
# 3. MAE, Huber, Quantile, MAPE, SMAPE, WeightedMSE Loss Tests
# =========================================================================


def _make_model_out_batch(predictions=None, targets=None):
    preds = predictions if predictions is not None else torch.tensor([[[1.0], [3.0]]])
    targs = targets if targets is not None else torch.tensor([[[2.0], [4.0]]])
    # MSELoss uses model_out['targets']; others use batch['targets']
    model_out = {"predictions": preds, "targets": targs, "other": preds}
    batch = {"targets": targs}
    return model_out, batch


def test_mae_loss_correctness():
    """MAE: |pred - target|, uses batch['targets']."""
    loss_fn = MAELoss(reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    # |1-2| + |3-4| = 1 + 1 = 2, mean = 1.0
    assert torch.isclose(result["loss"], torch.tensor(1.0))
    assert "other" in result


def test_mae_loss_reduction_sum():
    loss_fn = MAELoss(reduction="sum")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert torch.isclose(result["loss"], torch.tensor(2.0))


def test_huber_loss_forward():
    loss_fn = HuberLoss(delta=1.0, reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].dim() == 0
    assert result["loss"].item() > 0


def test_huber_loss_delta():
    loss_fn = HuberLoss(delta=0.5, reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert result["loss"].item() > 0


def test_quantile_loss_median():
    """Quantile 0.5 approximates MAE behavior."""
    loss_fn = QuantileLoss(quantile=0.5, reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() > 0


def test_quantile_loss_quantile_025():
    loss_fn = QuantileLoss(quantile=0.25, reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert result["loss"].item() >= 0


def test_quantile_loss_reduction_sum():
    loss_fn = QuantileLoss(quantile=0.5, reduction="sum")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert result["loss"].dim() == 0


def test_quantile_loss_invalid_raises():
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        QuantileLoss(quantile=0.0)
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        QuantileLoss(quantile=1.0)


def test_mape_loss_forward():
    loss_fn = MAPELoss(epsilon=1e-8, reduction="mean")
    model_out, batch = _make_model_out_batch(
        predictions=torch.tensor([[[2.0], [6.0]]]),
        targets=torch.tensor([[[2.0], [4.0]]]),
    )
    result = loss_fn(model_out, batch)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() >= 0


def test_mape_loss_epsilon_avoids_divzero():
    """Small targets get epsilon treatment to avoid div by zero."""
    loss_fn = MAPELoss(epsilon=1e-2, reduction="mean")
    model_out, batch = _make_model_out_batch(
        predictions=torch.tensor([[[0.1], [0.2]]]),
        targets=torch.tensor([[[1e-10], [1e-10]]]),
    )
    result = loss_fn(model_out, batch)
    assert not torch.isnan(result["loss"])
    assert not torch.isinf(result["loss"])


def test_smape_loss_forward():
    loss_fn = SMAPELoss(epsilon=1e-8, reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() >= 0
    assert result["loss"].item() <= 2.0  # SMAPE is bounded


def test_smape_loss_reduction_none():
    loss_fn = SMAPELoss(epsilon=1e-8, reduction="none")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert result["loss"].shape == model_out["predictions"].shape


def test_weighted_mse_loss_uniform_weights():
    """Without weights, should behave like MSE (weights=1)."""
    loss_fn = WeightedMSELoss(reduction="mean")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    # MSE: (1-2)^2 + (3-4)^2 = 1+1=2, mean=1.0
    assert torch.isclose(result["loss"], torch.tensor(1.0))


def test_weighted_mse_loss_explicit_weights():
    loss_fn = WeightedMSELoss(reduction="mean")
    model_out = {"predictions": torch.tensor([[[1.0], [3.0]]])}
    batch = {
        "targets": torch.tensor([[[2.0], [4.0]]]),
        "weights": torch.tensor([[[2.0], [1.0]]]),
    }
    result = loss_fn(model_out, batch)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].item() > 0


def test_weighted_mse_loss_reduction_sum():
    loss_fn = WeightedMSELoss(reduction="sum")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert result["loss"].dim() == 0


def test_weighted_mse_loss_reduction_none():
    loss_fn = WeightedMSELoss(reduction="none")
    model_out, batch = _make_model_out_batch()
    result = loss_fn(model_out, batch)
    assert result["loss"].shape == model_out["predictions"].shape


def test_combined_loss_forward():
    mae = MAELoss(reduction="mean")
    mse = MSELoss(reduction="mean")
    combined = CombinedLoss({"mae": mae, "mse": mse})
    model_out, batch = _make_model_out_batch()
    result = combined(model_out, batch)
    assert "loss" in result
    assert "mae_loss" in result
    assert "mse_loss" in result
    assert torch.isclose(result["loss"], (result["mae_loss"] + result["mse_loss"]) / 2)


def test_combined_loss_with_weights():
    mae = MAELoss(reduction="mean")
    mse = MSELoss(reduction="mean")
    combined = CombinedLoss({"mae": mae, "mse": mse}, weights={"mae": 2.0, "mse": 1.0})
    model_out, batch = _make_model_out_batch()
    result = combined(model_out, batch)
    total = 2.0 * result["mae_loss"] + 1.0 * result["mse_loss"]
    expected = total / 3.0  # normalized
    assert torch.isclose(result["loss"], expected)
