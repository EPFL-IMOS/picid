import torch.nn as nn


class SESModelBaseline(nn.Module):
    """
    Simple Exponential Smoothing (SES) baseline model for time series forecasting.

    This model implements exponential smoothing where recent observations are
    weighted more heavily using a smoothing parameter alpha.
    Formula: S_t = α * y_t + (1 - α) * S_{t-1}

    Parameters
    ----------
    pred_len : int
        Number of time steps to forecast ahead.
    alpha : float, optional
        Smoothing parameter (0 < alpha <= 1). Higher values give more weight
        to recent observations. Default is 0.3.
    **kwargs : Any
        Additional arguments for compatibility.
    """

    def __init__(self, pred_len, alpha=0.3, **kwargs):
        super().__init__()
        self.forecast_horizon = pred_len
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass for the SES baseline model.

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

        # Initialize smoothed value with first observation
        smoothed = x[:, 0, :].clone()  # (batch_size, num_features)

        # Apply exponential smoothing through the sequence
        for t in range(1, sequence_length):
            smoothed = self.alpha * x[:, t, :] + (1 - self.alpha) * smoothed

        # For SES, prediction is constant (the last smoothed value)
        # Repeat for all forecast steps
        predictions = smoothed.unsqueeze(1).repeat(1, self.forecast_horizon, 1)

        return predictions

    def __repr__(self):
        return f"SESModelBaseline(forecast_horizon={self.forecast_horizon}, alpha={self.alpha})"
