import numpy as np
import torch
from einops import rearrange
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from lmetk.model.wrappers.base import FitPredictWrapper


class FitPredictXGBoostWrapper(FitPredictWrapper):
    r"""Wrapper for the XGboost model."""

    def __init__(self, **kwargs):
        backbone = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, verbose=3)
        )
        super().__init__(backbone=backbone, **kwargs)

    def batch_to_sklearn(self, batch):

        forward_pattern = "samples sequence features -> samples (features sequence)"

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
        return X_fit, y_fit

    def sklearn_out_to_batch(self, outputs):
        r"""Convert sklearn outputs to batch format.

        Parameters
        ----------
        outputs : np.ndarray
            The outputs from the sklearn model.

        Returns
        -------
        dict
            A dictionary containing the batched data.
        """
        forward_pattern = "samples (features sequence) -> samples sequence features"

        out = rearrange(outputs, forward_pattern)
        return out
