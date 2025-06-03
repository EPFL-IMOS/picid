"""
Transformer-specific evaluator implementation for time series forecasting.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from .base import AbstractEvaluator
from lmetk.metrics.metric_factory import MetricFactory
from lightning_fabric.utilities.data import AttributeDict

class DefaultEvaluator(AbstractEvaluator):
    """Default evaluator that updates metrics incrementally."""
    
    """Sequential evaluator that updates metrics incrementally."""
    
    def __init__(self, 
                 metric_names: List[str], 
                 scaler: Optional[Any] = None, 
                 apply_inverse_scaling: bool = False,
                 **kwargs
                ):
        """
        Initialize the evaluator with specified metrics and optional scaling.
        
        Parameters
        ----------
        metric_names : List[str]
            List of metric names to compute.
        scaler : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling before computing metrics
        """
        super().__init__(scaler, apply_inverse_scaling, **kwargs)
        
        # Validate and create metrics
        self.metrics = {}
        for name in metric_names:
            metric = MetricFactory.create_metric(name)
            self.metrics[name.lower()] = metric
        
        self.metric_names = [name.lower() for name in metric_names]
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, model_out: dict):
        """
        Update metrics with new predictions and targets.
        
        Parameters
        ----------
        model_out : dict
            Dictionary containing 'predictions' and 'targets' keys with numpy arrays
        """
        predictions = np.asarray(model_out['predictions'])
        targets = np.asarray(model_out['targets'])
        
        if predictions.shape != targets.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
        
        # Apply inverse scaling if configured
        predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(predictions, targets)
        
        # Update all metrics
        for metric in self.metrics.values():
            metric.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metric values.
        
        Returns
        -------
        Dict[str, float]
            Dictionary with metric names as keys and computed values as values
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        
        return results