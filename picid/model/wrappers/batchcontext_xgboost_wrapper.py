import numpy as np
import torch
from einops import rearrange
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from lmetk.baselines.definitions import REGRESSION_TASKS, FORECASTING_TASKS
from lmetk.model.wrappers.base import AbstractFeedForwardWrapper


class BatchContextXGBoostWrapper(AbstractFeedForwardWrapper):
    """Wrapper for the XGboost model."""

    def __init__(self, task_type: str, **kwargs):
        # Define supported tasks based on your new categories
        self.regression_tasks = FORECASTING_TASKS
        self.classification_tasks = []
        self.supported_types = self.regression_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for BatchContextXGBoostWrapper.")

        self.task_type = task_type

        backbone = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, verbose=3)
        )
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0
        self.aux_context = None
        self.y_aux = None

    def fit(self, x, y):
        self.backbone.fit(x, y)

    def predict(self, x):
        return self.backbone.predict(x)

    def forward(
        self,
        batch,
    ):
        """Forward pass for the Tune wrapper. Note the following:
        future_context_timestamps := batch['context']['y_mark'] is not used due to possible noise,
        historical_target_timestamps := batch['target']['x_mark']  is redundant with historical_context_timestamps,
        future_target_timestamps := batch['target']['y_mark']  is redundant with future_context_timestamps.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        forward_pattern = "samples sequence features -> samples (features sequence)"
        inverse_pattern = "samples (features sequence) -> samples sequence features"

        historical_context_timestamps = rearrange(
            batch["context"]["x_mark"], forward_pattern
        )

        historical_context = rearrange(batch["context"]["x"], forward_pattern)
        future_context = rearrange(batch["context"]["y"], forward_pattern)

        historical_target = rearrange(batch["target"]["x"], forward_pattern)
        future_target = rearrange(batch["target"]["y"], forward_pattern)

        feature_list = [
            historical_context_timestamps,
            historical_context,
            historical_target,
            future_context,
        ]
        X_fit = np.nan_to_num(torch.concatenate(feature_list, axis=1).cpu().numpy())
        y_fit = future_target.cpu().numpy()

        # Fit model
        self.fit(X_fit, y_fit)

        # Predict
        outputs = self.predict(X_fit)

        # Update evaluator with current batch
        n_sequence, n_features = (
            batch["target"]["y"].shape[1],
            batch["target"]["y"].shape[2],
        )
        model_out = {
            "predictions": rearrange(
                torch.Tensor(outputs),
                inverse_pattern,
                sequence=n_sequence,
                features=n_features,
            ),
            "targets": batch["target"]["y"],
        }

        return model_out
