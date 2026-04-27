import torch
import torch.nn as nn


class LinearRegressionModelBaseline(nn.Module):
    """
    Linear Regression baseline model for time series forecasting.

    This model uses the last k values as features to predict future values
    using linear regression. Each feature (channel) is modeled independently.

    Parameters
    ----------
    pred_len : int
        Number of time steps to forecast ahead.
    lag_features : int, optional
        Number of lagged values to use as features. Default is 5.
    **kwargs : Any
        Additional arguments for compatibility.
    """

    def __init__(self, pred_len, lag_features, **kwargs):
        super().__init__()
        self.forecast_horizon = pred_len
        self.lag_features = lag_features

        # Note: We'll fit the linear regression parameters during forward pass
        # since we don't have a separate training phase in this baseline setup

    def forward(self, x):
        """
        Forward pass for the linear regression baseline model.

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

        effective_lag = min(self.lag_features, sequence_length - 1)

        if effective_lag < 1:
            # Fall back to naive baseline if insufficient data
            last_values = x[:, -1:, :]
            return last_values.repeat(1, self.forecast_horizon, 1)

        predictions = []

        # Fit and predict for each feature independently
        for feature_idx in range(num_features):
            feature_data = x[:, :, feature_idx]  # (batch_size, sequence_length)
            feature_predictions = []

            # Process each sample in the batch
            for batch_idx in range(batch_size):
                sample_data = feature_data[batch_idx]  # (sequence_length,)

                # Create lagged features matrix
                X_features = []
                y_targets = []

                for t in range(effective_lag, sequence_length):
                    # Use last effective_lag values as features
                    features = sample_data[t - effective_lag : t]
                    target = sample_data[t]
                    X_features.append(features)
                    y_targets.append(target)

                if len(X_features) == 0:
                    # Not enough data, use last value
                    pred = sample_data[-1].repeat(self.forecast_horizon)
                else:
                    X_features = torch.stack(X_features)  # (n_samples, effective_lag)
                    y_targets = torch.stack(y_targets)  # (n_samples,)

                    # Solve normal equations: theta = (X^T X)^{-1} X^T y
                    # Add bias term
                    X_with_bias = torch.cat(
                        [
                            torch.ones(X_features.shape[0], 1, device=x.device),
                            X_features,
                        ],
                        dim=1,
                    )

                    # Use least squares solution
                    XtX = torch.matmul(X_with_bias.T, X_with_bias)
                    Xty = torch.matmul(X_with_bias.T, y_targets)

                    # Add small regularization for numerical stability
                    reg = 1e-6 * torch.eye(XtX.shape[0], device=x.device)
                    theta = torch.linalg.solve(XtX + reg, Xty)

                    # Generate predictions
                    pred = []
                    current_features = sample_data[-effective_lag:].clone()

                    for h in range(self.forecast_horizon):
                        # Prepare features with bias
                        features_with_bias = torch.cat(
                            [torch.ones(1, device=x.device), current_features]
                        )

                        # Predict next value
                        next_pred = torch.dot(theta, features_with_bias)
                        pred.append(next_pred)

                        # Update features for next prediction (shift window)
                        current_features = torch.cat(
                            [current_features[1:], next_pred.unsqueeze(0)]
                        )

                    pred = torch.stack(pred)

                feature_predictions.append(pred)

            # Stack predictions for all samples in batch
            feature_predictions = torch.stack(
                feature_predictions
            )  # (batch_size, forecast_horizon)
            predictions.append(feature_predictions)

        # Stack predictions for all features
        predictions = torch.stack(
            predictions, dim=2
        )  # (batch_size, forecast_horizon, num_features)

        return predictions

    def __repr__(self):
        return f"LinearRegressionModelBaseline(forecast_horizon={self.forecast_horizon}, lag_features={self.lag_features})"
