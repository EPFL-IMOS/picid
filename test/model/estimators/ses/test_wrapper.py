import torch
import pytest

from picid.model.estimators.ses.wrapper import SESModelWrapper


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


def test_ses_model_wrapper_initialization():
    wrapper = SESModelWrapper(
        task_type="forecasting",
        pred_len=3,
        alpha=0.3,
    )
    assert wrapper.task_type == "forecasting"
    assert wrapper.kwargs.pred_len == 3


def test_ses_model_wrapper_invalid_task_type_raises():
    with pytest.raises(ValueError, match="Task regression not supported"):
        SESModelWrapper(
            task_type="regression",
            pred_len=3,
            alpha=0.3,
        )


def test_ses_model_wrapper_forward_shape():
    batch_size, seq_len, pred_len, num_features = 4, 10, 3, 2
    batch = _make_batch(batch_size, seq_len, pred_len, num_features)

    wrapper = SESModelWrapper(
        task_type="forecasting",
        pred_len=pred_len,
        alpha=0.3,
    )

    model_out = wrapper(batch)

    assert "predictions" in model_out
    assert "targets" in model_out
    assert model_out["predictions"].shape == (batch_size, pred_len, num_features)
    assert model_out["targets"].shape == (batch_size, pred_len, num_features)


def test_ses_model_wrapper_forward_with_custom_alpha():
    batch = _make_batch(batch_size=2, seq_len=5, pred_len=2, num_features=1)

    wrapper = SESModelWrapper(
        task_type="forecasting",
        pred_len=2,
        alpha=0.5,
    )

    model_out = wrapper(batch)

    assert model_out["predictions"].shape == (2, 2, 1)
    # With alpha=0.5, SES gives more weight to recent values
    assert not torch.isnan(model_out["predictions"]).any()
    assert not torch.isinf(model_out["predictions"]).any()
