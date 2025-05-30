"""Wrapper for the LinearRegression model."""
import torch
import numpy as np
from .base import AbstractWrapper

from lmetk.model.methods.linear_regression_model import LinearRegressionModelBaseline
from lightning_fabric.utilities.data import AttributeDict

class LinearRegressionModelWrapper(AbstractWrapper):
    r"""Wrapper for the LinearRegressionModel baseline.

    This wrapper defines the forward pass of the LinearRegressionModel baseline.
    """
    def __init__(self, **kwargs):
        backbone = LinearRegressionModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        r"""Forward pass for the LinearRegression wrapper.

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
        f_dim = -1 if self.kwargs.features_mode == 'MS' else 0
        outputs = outputs[:, -self.kwargs.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.kwargs.pred_len:, f_dim:]
        
        outputs = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()
            
        model_out = {
            'predictions': outputs,
            'targets': batch_y
        }
        
        return model_out