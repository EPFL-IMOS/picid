import torch

from picid.model.estimators.linear_regression.model import LinearRegressionModelBaseline


def test_linear_regression_model_reports_canonical_module():
    assert (
        LinearRegressionModelBaseline.__module__
        == "picid.model.estimators.linear_regression.model"
    )


def test_linear_regression_model_forward_output_shape():
    batch_size, seq_len, pred_len, num_features = 3, 10, 3, 2
    model = LinearRegressionModelBaseline(pred_len=pred_len, lag_features=3)

    x = torch.randn(batch_size, seq_len, num_features)
    out = model(x)

    assert out.shape == (batch_size, pred_len, num_features)


def test_linear_regression_model_effective_lag_fallback_short_sequence():
    """When sequence_length is 1, effective_lag is 0: repeat last value (naive fallback)."""
    pred_len = 4
    model = LinearRegressionModelBaseline(pred_len=pred_len, lag_features=5)

    x = torch.tensor([[[2.0], [99.0]]])[
        :, :1, :
    ]  # (1, 1, 1) — only first timestep kept
    assert x.shape == (1, 1, 1)

    out = model(x)

    assert out.shape == (1, pred_len, 1)
    assert torch.allclose(out, torch.full((1, pred_len, 1), 2.0))


def test_linear_regression_model_least_squares_linear_trend():
    """Perfect linear series: next-step coefficients recover slope 1, intercept 0."""
    pred_len = 2
    lag_features = 1
    model = LinearRegressionModelBaseline(pred_len=pred_len, lag_features=lag_features)

    # Values 0,1,2,3 -> from last lag [2] predict 3, then roll with predictions
    x = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    out = model(x)

    assert out.shape == (1, pred_len, 1)
    assert torch.allclose(out[0, 0, 0], torch.tensor(4.0), atol=1e-4)
    assert torch.allclose(out[0, 1, 0], torch.tensor(5.0), atol=1e-4)


def test_linear_regression_model_multi_feature_independent_channels():
    """Each feature column is regressed independently; shapes are (B, H, C)."""
    pred_len = 2
    lag_features = 1
    model = LinearRegressionModelBaseline(pred_len=pred_len, lag_features=lag_features)

    # Channel 0: unit slope. Channel 1: slope 2 — same length, different OLS coefficients.
    x = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 6.0],
            ]
        ]
    )
    out = model(x)

    assert out.shape == (1, pred_len, 2)
    assert torch.allclose(out[0, :, 0], torch.tensor([4.0, 5.0]), atol=1e-4)
    assert torch.allclose(out[0, :, 1], torch.tensor([8.0, 10.0]), atol=1e-4)


def test_linear_regression_model_repr():
    model = LinearRegressionModelBaseline(pred_len=7, lag_features=4)
    repr_str = repr(model)
    assert "LinearRegressionModelBaseline" in repr_str
    assert "7" in repr_str
    assert "4" in repr_str
