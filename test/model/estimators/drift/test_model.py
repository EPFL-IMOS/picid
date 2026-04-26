import torch

from picid.model.estimators.drift.model import DriftModelBaseline


def test_drift_model_reports_canonical_module():
    assert DriftModelBaseline.__module__ == "picid.model.estimators.drift.model"


def test_drift_model_forward_output_shape():
    """Forward returns (batch_size, forecast_horizon, num_features)."""
    batch_size, seq_len, pred_len, num_features = 4, 10, 3, 2
    model = DriftModelBaseline(pred_len=pred_len)

    x = torch.randn(batch_size, seq_len, num_features)
    out = model(x)

    assert out.shape == (batch_size, pred_len, num_features)


def test_drift_model_linear_extrapolation():
    """Drift extrapolates linearly from first to last point."""
    model = DriftModelBaseline(pred_len=3)

    # x: first=0, last=3, drift per step = (3-0)/(3-1) = 1.5
    # pred at h=1: 3 + 1*1.5 = 4.5, h=2: 6.0, h=3: 7.5
    x = torch.tensor([[[0.0], [1.5], [3.0]]])  # (1, 3, 1)
    out = model(x)

    assert out.shape == (1, 3, 1)
    assert torch.allclose(out[0, 0, 0], torch.tensor(4.5))
    assert torch.allclose(out[0, 1, 0], torch.tensor(6.0))
    assert torch.allclose(out[0, 2, 0], torch.tensor(7.5))


def test_drift_model_zero_drift():
    """When first==last, drift is zero, predictions are constant."""
    model = DriftModelBaseline(pred_len=4)

    x = torch.tensor([[[5.0], [5.0], [5.0]]])  # (1, 3, 1)
    out = model(x)

    assert out.shape == (1, 4, 1)
    assert torch.allclose(out, torch.full_like(out, 5.0))


def test_drift_model_short_sequence_fallback():
    """Sequence length < 2 falls back to naive (repeat last value)."""
    model = DriftModelBaseline(pred_len=3)

    x = torch.tensor([[[7.0]]])  # (1, 1, 1)
    out = model(x)

    assert out.shape == (1, 3, 1)
    assert torch.allclose(out, torch.full_like(out, 7.0))


def test_drift_model_repr():
    model = DriftModelBaseline(pred_len=7)
    repr_str = repr(model)
    assert "7" in repr_str
