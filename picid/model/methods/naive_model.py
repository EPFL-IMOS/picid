import torch.nn as nn


class NaiveModelBaseline(nn.Module):
    """Naive baseline model for time series forecasting.

    This model implements a flexible baseline where predictions can be based on:
    - The last observed value (window_size=1): y_{t+1} = y_t
    - An average of the last N observations (window_size=N): y_{t+1} = mean(y_{t-N+1:t})
    For multi-step forecasting, it repeats the computed value.

    Parameters
    ----------
    pred_len : int, optional
        Number of time steps to forecast ahead. Default is 1.
    window_size_to_average : int, optional
        Number of previous observations to average. If 1, uses the last value only.
        If > 1, uses the mean of the last window_size observations. Default is 1.
    """

    def __init__(self, pred_len, window_size_to_average, **kwargs):
        super().__init__()
        self.forecast_horizon = pred_len
        self.window_size = window_size_to_average

    def forward(self, x):
        """Forward pass for the naive baseline model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_features)

        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch_size, forecast_horizon, out_channels)
        """
        # x shape: (batch_size, sequence_length, num_features)
        batch_size, sequence_length, num_features = x.shape

        # Ensure we don't try to use more observations than available
        effective_window = min(self.window_size, sequence_length)

        # Get the last window_size observations
        # Shape: (batch_size, effective_window, out_channels)
        last_window = x[:, -effective_window:, :]

        # Compute the mean over the window dimension
        # Shape: (batch_size, 1, out_channels)
        baseline_values = last_window.mean(dim=1, keepdim=True)

        # Repeat the baseline values for the forecast horizon
        predictions = baseline_values.repeat(1, self.forecast_horizon, 1)

        return predictions

    def __repr__(self):
        return f"NaiveBaseline(forecast_horizon={self.forecast_horizon}, window_size={self.window_size})"
