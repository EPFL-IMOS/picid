import torch
import torch.nn as nn


class DriftModelBaseline(nn.Module):
    """
    Drift baseline model for time series forecasting.

    This model implements linear extrapolation based on the average change
    between the first and last point in the input sequence.
    Formula: y_{t+h} = y_t + h * ((y_t - y_1) / (t - 1))

    Parameters
    ----------
    pred_len : int
        Number of time steps to forecast ahead.
    **kwargs : Any
        Additional arguments for compatibility.
    """

    def __init__(self, pred_len, **kwargs):
        super().__init__()
        self.forecast_horizon = pred_len

    def forward(self, x):
        """
        Forward pass for the drift baseline model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_features).

        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch_size, forecast_horizon, num_features).
        """
        # x shape: (batch_size, sequence_length, num_features)
        batch_size, sequence_length, num_features = x.shape

        if sequence_length < 2:
            # If sequence too short, fall back to naive baseline
            last_values = x[:, -1:, :]
            return last_values.repeat(1, self.forecast_horizon, 1)

        # Get first and last observations
        first_obs = x[:, 0, :]  # (batch_size, num_features)
        last_obs = x[:, -1, :]  # (batch_size, num_features)

        # Calculate drift (average change per time step)
        drift = (last_obs - first_obs) / (sequence_length - 1)

        # Generate predictions for each forecast step
        predictions = []
        for h in range(1, self.forecast_horizon + 1):
            pred_step = last_obs + h * drift
            predictions.append(pred_step.unsqueeze(1))

        # Stack predictions: (batch_size, forecast_horizon, num_features)
        predictions = torch.cat(predictions, dim=1)

        return predictions

    def __repr__(self):
        return f"DriftModelBaseline(forecast_horizon={self.forecast_horizon})"
