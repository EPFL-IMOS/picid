"""Window-average backbone."""

import torch.nn as nn


class WindowAverageBaseline(nn.Module):
    """
    Window-average baseline model for time series forecasting.

    This model implements a flexible baseline where predictions can be based on:
    - The last observed value (window_size=1): y_{t+1} = y_t
    - An average of the last N observations (window_size=N): y_{t+1} = mean(y_{t-N+1:t})
    For multi-step forecasting, it repeats the computed value.

    Parameters
    ----------
    pred_len : int
        Number of time steps to forecast ahead.
    window_size_to_average : int
        Number of trailing observations to average.
    **kwargs : Any
        Additional compatibility arguments accepted but ignored.
    """

    def __init__(self, pred_len, window_size_to_average, **kwargs):
        """
        Initialize the window-average baseline.

        Parameters
        ----------
        pred_len : int
            Number of time steps to forecast ahead.
        window_size_to_average : int
            Number of trailing observations to average.
        **kwargs : Any
            Additional compatibility arguments accepted but ignored.
        """
        super().__init__()
        self.forecast_horizon = pred_len
        self.window_size = window_size_to_average

    def forward(self, x):
        """
        Forecast by repeating the trailing-window average.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, sequence_length, num_features)``.

        Returns
        -------
        torch.Tensor
            Forecast tensor of shape ``(batch_size, pred_len, num_features)``.
        """
        batch_size, sequence_length, num_features = x.shape
        effective_window = min(self.window_size, sequence_length)
        last_window = x[:, -effective_window:, :]
        baseline_values = last_window.mean(dim=1, keepdim=True)
        return baseline_values.repeat(1, self.forecast_horizon, 1)

    def __repr__(self):
        return (
            "WindowAverageBaseline("
            f"forecast_horizon={self.forecast_horizon}, "
            f"window_size={self.window_size})"
        )


__all__ = ["WindowAverageBaseline"]
