"""Wrapper for the LinearRegression model."""
import torch
import numpy as np
from .base import AbstractWrapper

from lmetk.model.methods.drift_model import DriftModelBaseline
from lightning_fabric.utilities.data import AttributeDict


class DriftModelWrapper(AbstractWrapper):
    r"""Wrapper for the DriftModel baseline.

    This wrapper defines the forward pass of the DriftModel baseline.
    """
    def __init__(self, **kwargs):
        backbone = DriftModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        r"""Forward pass for the Drift wrapper.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        batch_x = batch['batch_x']
        batch_y = batch['batch_y']
        
        outputs = self.backbone(batch_x)
            
        # Process outputs following your pattern
        outputs = outputs[:, -self.kwargs.pred_len:, :]
        batch_y = batch_y[:, -self.kwargs.pred_len:, :]
            
        model_out = {
            'predictions': outputs,
            'targets': batch_y
        }
        
        return model_out