import torch
import pytest

from picid.model.estimators.linear_regression.wrapper import LinearRegressionModelWrapper


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


def test_linear_regression_model_wrapper_initialization():
    wrapper = LinearRegressionModelWrapper(
        task_type="forecasting",
        pred_len=3,
        lag_features=2,
    )
    assert wrapper.task_type == "forecasting"
    assert wrapper.kwargs.pred_len == 3
    assert wrapper.kwargs.lag_features == 2


def test_linear_regression_model_wrapper_invalid_task_type_raises():
    with pytest.raises(
        ValueError,
        match="Task regression not supported for LinearRegressionModelWrapper",
    ):
        LinearRegressionModelWrapper(
            task_type="regression",
            pred_len=3,
            lag_features=2,
        )


def test_linear_regression_model_wrapper_forward_shape_and_slicing():
    batch_size, seq_len, pred_len, num_features = 3, 12, 4, 3
    batch = _make_batch(batch_size, seq_len, pred_len, num_features)

    wrapper = LinearRegressionModelWrapper(
        task_type="forecasting",
        pred_len=pred_len,
        lag_features=3,
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


def test_linear_regression_model_wrapper_multi_feature_forward():
    """Several output channels; wrapper preserves (B, pred_len, C)."""
    pred_len, num_features = 2, 4
    batch = {
        "target": {
            "target_seq_x": torch.arange(24, dtype=torch.float32).reshape(1, 6, 4),
            "target_seq_y": torch.zeros(1, 8, 4),
        }
    }
    wrapper = LinearRegressionModelWrapper(
        task_type="forecasting",
        pred_len=pred_len,
        lag_features=2,
    )
    out = wrapper(batch)
    assert out["predictions"].shape == (1, pred_len, num_features)
    assert not torch.isnan(out["predictions"]).any()
