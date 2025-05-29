"""Utilities for handling data scaling and inverse scaling."""

import numpy as np
from typing import Optional, Any


class ScalingWrapper:
    """Wrapper class to handle inverse scaling operations."""
    
    def __init__(self, scaler: Optional[Any] = None, apply_inverse_scaling: bool = False):
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
    
    def inverse_transform_if_needed(self, predictions: np.ndarray, targets: np.ndarray) -> tuple:
        """
        Apply inverse transform to predictions and targets if needed.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        targets : np.ndarray
            Ground truth targets
            
        Returns
        -------
        tuple
            (transformed_predictions, transformed_targets)
        """
        if not (self.scaler and self.apply_inverse):
            return predictions, targets
        
        # Apply inverse transform while preserving original shapes
        pred_shape = predictions.shape
        target_shape = targets.shape
        
        # Apply inverse transform (assuming scaler expects 2D input)
        predictions_transformed = self.scaler.inverse_transform(
            predictions.squeeze(0)
        ).reshape(pred_shape)
        
        targets_transformed = self.scaler.inverse_transform(
            targets.squeeze(0)
        ).reshape(target_shape)
        
        return predictions_transformed, targets_transformed