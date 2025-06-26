"""Wrapper for the TimePFN model."""
import torch
from .base import AbstractWrapper

from lmetk.model.time_pfn.TimePFN import TimePFN
from lightning_fabric.utilities.data import AttributeDict
from autogluon.tabular import TabularPredictor

import numpy as np
import pandas as pd


def dataloader_to_dataframe(dataloader):
    X_list, y_list = [], []
    for X, y in dataloader:
        X_list.append(X.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())
    X_np = np.concatenate(X_list, axis=0)
    y_np = np.concatenate(y_list, axis=0)
    df = pd.DataFrame(X_np)
    df['target'] = y_np
    return df




class TabPFNWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """
    def __init__(self, **kwargs):
        backbone = TabularPredictor(label="target")
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0

        self.train_dataloader = kwargs['train_dataloader']
        self.val_dataloader = kwargs['val_dataloader']
        
        self.aux_context = None
        self.y_aux = None
    
    def fit(self, x, y):
        self.backbone.fit(x.cpu(), y.cpu())
    
    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        context_fit = batch['context']['x'].squeeze(0)
        target_fit = batch['target']['x'].squeeze(0)

        context = batch['context']['y'].squeeze(0)
        target = batch['target']['y']
        
        # Extent the context with auxiliary features to give sence of seasonality
        self.fit(context_fit, target_fit)

        # Extent the context with auxiliary features to give sence of seasonality
        outputs = self.backbone.predict(context.cpu())
            
        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            'predictions': torch.Tensor(outputs).reshape(batch['target']['y'].shape),
            'targets': target
        }

        
        return model_out