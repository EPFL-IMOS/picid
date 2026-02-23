import torch
import pytest

from picid.model.wrappers.naive_model_wrapper import NaiveModelWrapper


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


def test_naive_model_wrapper_initialization():
    wrapper = NaiveModelWrapper(
        task_type="forecasting",
        pred_len=3,
        window_size_to_average=1,
    )
    assert wrapper.task_type == "forecasting"
    assert wrapper.kwargs.pred_len == 3


def test_naive_model_wrapper_invalid_task_type_raises():
    with pytest.raises(ValueError, match="Task regression not supported"):
        NaiveModelWrapper(
            task_type="regression",
            pred_len=3,
            window_size_to_average=1,
        )


def test_naive_model_wrapper_forward_shape():
    batch_size, seq_len, pred_len, num_features = 4, 10, 3, 2
    batch = _make_batch(batch_size, seq_len, pred_len, num_features)

    wrapper = NaiveModelWrapper(
        task_type="forecasting",
        pred_len=pred_len,
        window_size_to_average=1,
    )

    model_out = wrapper(batch)

    assert "predictions" in model_out
    assert "targets" in model_out
    assert model_out["predictions"].shape == (batch_size, pred_len, num_features)
    assert model_out["targets"].shape == (batch_size, pred_len, num_features)


def test_naive_model_wrapper_forward_with_window_average():
    batch = _make_batch(batch_size=2, seq_len=8, pred_len=2, num_features=1)

    wrapper = NaiveModelWrapper(
        task_type="forecasting",
        pred_len=2,
        window_size_to_average=4,
    )

    model_out = wrapper(batch)

    assert model_out["predictions"].shape == (2, 2, 1)
    # Predictions should be mean of last 4 values, repeated for pred_len
    expected_mean = batch["target"]["target_seq_x"][:, -4:, :].mean(dim=1, keepdim=True)
    assert torch.allclose(
        model_out["predictions"][:, 0:1, :], expected_mean, atol=1e-5
    )
    assert torch.allclose(
        model_out["predictions"][:, 1:2, :], expected_mean, atol=1e-5
    )
