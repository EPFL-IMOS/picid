"""Utilities for handling data scaling and inverse scaling."""

import torch
import numpy as np
from typing import Optional, Any, Tuple


class ScalingWrapper:
    """Wrapper class to handle inverse scaling operations."""
    
    def __init__(self, scaler: Optional[Any] = None, apply_inverse_scaling: bool = False, task_mode: bool = False):
        """
        Initialize the scaling wrapper.
        
        Parameters
        ----------
        scaler : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling
        """
        self.scaler = scaler
        self.apply_inverse = apply_inverse_scaling
        self.task_mode = task_mode
    
    def inverse_transform_if_needed(self, 
                                    predictions: Tuple[torch.Tensor, np.ndarray], 
                                    targets: Tuple[torch.Tensor, np.ndarray]) -> tuple:
        """
        Apply inverse transform to predictions and targets if needed.
        
        Parameters
        ----------
        predictions : np.ndarray or torch.Tensor
            Model predictions
        targets : np.ndarray or torch.Tensor
            Ground truth targets
            
        Returns
        -------
        tuple
            (transformed_predictions, transformed_targets)
        """
        if not (self.scaler and self.apply_inverse):
            return predictions, targets
        
        # Check if the incomming data is torch tensor, is so, then covert to numpy 
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().detach().numpy()

        # Apply inverse transform while preserving original shapes
        pred_shape = predictions.shape
        target_shape = targets.shape
        
        # IT IS A PATCH: we should define the correct shate of input of the metrics! TABFPM produces [N_pred:NUM VAR TO PREDICT], TIMEPFN prodcuces: [1, N_pred, NUM VAR TO PREDICT]
        if len(pred_shape)==3:
            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.scaler.inverse_transform(
                predictions.squeeze()
            ).reshape(pred_shape)
            
            targets_transformed = self.scaler.inverse_transform(
                targets.squeeze()
            ).reshape(target_shape)
        
        elif len(pred_shape)==2:
            # Apply inverse transform (assuming scaler expects 2D input)
            predictions_transformed = self.scaler.inverse_transform(predictions
            ).reshape((1,-1))
            
            targets_transformed = self.scaler.inverse_transform(targets
            ).reshape((1,-1))
        else:
            raise ValueError(f"Unexpected shape: {pred_shape}")
        
        return predictions_transformed, targets_transformed