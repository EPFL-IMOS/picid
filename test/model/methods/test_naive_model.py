import torch

from picid.model.methods.naive_model import NaiveModelBaseline


def test_naive_model_forward_output_shape():
    """Forward returns (batch_size, forecast_horizon, num_features)."""
    batch_size, seq_len, pred_len, num_features = 4, 10, 3, 2
    model = NaiveModelBaseline(pred_len=pred_len, window_size_to_average=1)

    x = torch.randn(batch_size, seq_len, num_features)
    out = model(x)

    assert out.shape == (batch_size, pred_len, num_features)


def test_naive_model_window_size_one_uses_last_value():
    """With window_size=1, prediction should equal last input value repeated."""
    model = NaiveModelBaseline(pred_len=4, window_size_to_average=1)

    x = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
    out = model(x)

    assert out.shape == (1, 4, 1)
    expected = torch.tensor([[[3.0], [3.0], [3.0], [3.0]]])
    assert torch.allclose(out, expected)


def test_naive_model_window_average():
    """With window_size>1, prediction should be mean of last window values."""
    model = NaiveModelBaseline(pred_len=2, window_size_to_average=3)

    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])  # last 3: 2,3,4 -> mean 3.0
    out = model(x)

    assert out.shape == (1, 2, 1)
    expected_mean = 3.0
    assert torch.allclose(out, torch.full_like(out, expected_mean))


def test_naive_model_repr():
    model = NaiveModelBaseline(pred_len=5, window_size_to_average=3)
    repr_str = repr(model)
    assert "5" in repr_str
    assert "3" in repr_str
