import torch

from picid.model.estimators.ses.model import SESModelBaseline


def test_ses_model_reports_canonical_module():
    assert SESModelBaseline.__module__ == "picid.model.estimators.ses.model"


def test_ses_model_forward_output_shape():
    """Forward returns (batch_size, forecast_horizon, num_features)."""
    batch_size, seq_len, pred_len, num_features = 4, 10, 3, 2
    model = SESModelBaseline(pred_len=pred_len, alpha=0.3)

    x = torch.randn(batch_size, seq_len, num_features)
    out = model(x)

    assert out.shape == (batch_size, pred_len, num_features)


def test_ses_model_single_step_equals_last_smoothed():
    """With seq_len=1, output equals input repeated."""
    model = SESModelBaseline(pred_len=3, alpha=0.5)

    x = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
    out = model(x)

    assert out.shape == (1, 3, 2)
    expected = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
    assert torch.allclose(out, expected)


def test_ses_model_constant_prediction_across_horizon():
    """SES produces constant forecast (same value for each step)."""
    model = SESModelBaseline(pred_len=5, alpha=0.3)

    x = torch.randn(2, 8, 3)
    out = model(x)

    assert out.shape == (2, 5, 3)
    # All steps in horizon should be identical per batch/sample
    for b in range(2):
        for f in range(3):
            assert torch.allclose(out[b, 0, f], out[b, :, f])


def test_ses_model_repr():
    model = SESModelBaseline(pred_len=5, alpha=0.25)
    repr_str = repr(model)
    assert "5" in repr_str
    assert "0.25" in repr_str
