import torch

from picid.model.estimators.window_average.wrapper import WindowAverageWrapper


def _make_batch(batch_size=2, seq_len=6, pred_len=2, num_features=1):
    target_seq_x = torch.randn(batch_size, seq_len, num_features)
    target_seq_y = torch.randn(batch_size, seq_len + pred_len, num_features)
    return {
        "target": {
            "target_seq_x": target_seq_x,
            "target_seq_y": target_seq_y,
        }
    }


def test_window_average_wrapper_is_canonical_named():
    WindowAverageWrapper(
        task_type="forecasting",
        pred_len=2,
        window_size_to_average=1,
    )

    assert WindowAverageWrapper.__name__ == "WindowAverageWrapper"
    assert (
        WindowAverageWrapper.__module__
        == "picid.model.estimators.window_average.wrapper"
    )


def test_window_average_wrapper_keeps_forecasting_behavior():
    batch = _make_batch()
    wrapper = WindowAverageWrapper(
        task_type="forecasting",
        pred_len=2,
        window_size_to_average=2,
    )

    out = wrapper(batch)

    assert out["predictions"].shape == (2, 2, 1)
