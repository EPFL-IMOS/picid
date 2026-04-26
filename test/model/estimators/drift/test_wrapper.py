import torch
import pytest

from picid.model.estimators.drift.wrapper import DriftModelWrapper


def _make_batch(batch_size=4, seq_len=10, pred_len=3, num_features=2):
    """Create a batch dict matching the wrapper's expected structure."""
    target_seq_x = torch.randn(batch_size, seq_len, num_features)
    target_seq_y = torch.randn(batch_size, seq_len + pred_len, num_features)
    return {
        "target": {
            "target_seq_x": target_seq_x,
            "target_seq_y": target_seq_y,
        }
    }


def test_drift_model_wrapper_initialization():
    wrapper = DriftModelWrapper(
        task_type="forecasting",
        pred_len=3,
    )
    assert wrapper.task_type == "forecasting"
    assert wrapper.kwargs.pred_len == 3


def test_drift_model_wrapper_invalid_task_type_raises():
    with pytest.raises(
        ValueError, match="Task regression not supported for DriftModelWrapper"
    ):
        DriftModelWrapper(
            task_type="regression",
            pred_len=3,
        )


def test_drift_model_wrapper_forward_shape_and_slicing():
    batch_size, seq_len, pred_len, num_features = 4, 10, 3, 2
    batch = _make_batch(batch_size, seq_len, pred_len, num_features)

    wrapper = DriftModelWrapper(
        task_type="forecasting",
        pred_len=pred_len,
    )

    model_out = wrapper(batch)

    assert "predictions" in model_out
    assert "targets" in model_out
    assert model_out["predictions"].shape == (batch_size, pred_len, num_features)
    assert model_out["targets"].shape == (batch_size, pred_len, num_features)

    raw_out = wrapper.backbone(batch["target"]["target_seq_x"])
    expected_pred = raw_out[:, -pred_len:, :]
    expected_y = batch["target"]["target_seq_y"][:, -pred_len:, :]
    assert torch.allclose(model_out["predictions"], expected_pred)
    assert torch.allclose(model_out["targets"], expected_y)


def test_drift_model_wrapper_forward_matches_backbone_on_toy_series():
    """Deterministic batch: wrapper output equals last pred_len slice of backbone output."""
    pred_len = 2
    batch = {
        "target": {
            "target_seq_x": torch.tensor([[[0.0], [1.0], [3.0]]]),
            "target_seq_y": torch.ones(1, 5, 1),
        }
    }
    wrapper = DriftModelWrapper(task_type="forecasting", pred_len=pred_len)
    out = wrapper(batch)
    backbone_out = wrapper.backbone(batch["target"]["target_seq_x"])

    assert torch.allclose(out["predictions"], backbone_out[:, -pred_len:, :])
    assert out["predictions"].shape == (1, pred_len, 1)
