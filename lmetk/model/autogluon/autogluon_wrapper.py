"""Wrapper for the TimePFN model."""
import torch
from lmetk.model.wrappers.base import AbstractWrapper


import numpy as np
import pandas as pd

from einops import rearrange

from lmetk.model.autogluon.multi_label_predictor import MultilabelTabularPredictor


class AutoGluonWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """
    def __init__(self, presets: str = "best", **kwargs):
        label_names = [f"target_t+{i}" for i in range(1, kwargs["pred_len"]+1)]
        backbone = MultilabelTabularPredictor(labels=label_names)
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0

        self.label_names = label_names
        self.presets = presets

        self.aux_context = None
        self.y_aux = None

    def fit(self, x, y):
        df = pd.DataFrame(x)
        df[self.label_names] = y
        self.backbone.fit(df, presets=self.presets)

    def predict(self, x):
        df = pd.DataFrame(x)
        return self.backbone.predict(df)

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper. Note the following:
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

        forward_pattern = 'samples sequence features -> samples (features sequence)'
        inverse_pattern = 'samples (features sequence) -> samples sequence features'

        historical_context_timestamps = rearrange(batch['context']['x_mark'], forward_pattern)

        historical_context = rearrange(batch['context']['x'], forward_pattern)
        future_context = rearrange(batch['context']['y'], forward_pattern)

        historical_target = rearrange(batch['target']['x'], forward_pattern)
        future_target = rearrange(batch['target']['y'], forward_pattern)

        feature_list = [historical_context_timestamps, historical_context, historical_target, future_context]
        X_fit = np.nan_to_num(torch.concatenate(feature_list, axis=1).cpu().numpy())
        y_fit = future_target.cpu().numpy()

        # Fit model
        self.fit(X_fit, y_fit)

        # Predict
        outputs = self.predict(X_fit)

        # Update evaluator with current batch
        n_sequence, n_features = batch['target']['y'].shape[1], batch['target']['y'].shape[2]
        model_out = {
            'predictions': rearrange(torch.Tensor(outputs), inverse_pattern, sequence=n_sequence, features=n_features),
            'targets': batch['target']['y']
        }

        
        return model_out