import torch

from picid.model.estimators.window_average.model import WindowAverageBaseline


def test_window_average_baseline_is_canonical_named():
    model = WindowAverageBaseline(pred_len=3, window_size_to_average=2)

    assert WindowAverageBaseline.__name__ == "WindowAverageBaseline"
    assert (
        WindowAverageBaseline.__module__
        == "picid.model.estimators.window_average.model"
    )
    assert "WindowAverageBaseline" in repr(model)


def test_window_average_baseline_keeps_forecast_behavior():
    model = WindowAverageBaseline(pred_len=2, window_size_to_average=3)

    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    out = model(x)

    expected = torch.tensor([[[3.0], [3.0]]])
    assert torch.allclose(out, expected)
