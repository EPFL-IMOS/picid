"""
Default loss implementations for time series transformer models.

This module provides concrete implementations of common loss functions
used in time series forecasting and analysis tasks.
"""

from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_loss import AbstractLoss


class MSELoss(AbstractLoss):
    """
    Mean Squared Error loss for time series regression tasks.
    
    This is the most common loss function for time series forecasting,
    penalizing squared differences between predictions and targets.
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initialize MSE loss.
        
        Args:
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MSE loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        
        loss = self.mse_loss(predictions, targets)
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class MAELoss(AbstractLoss):
    """
    Mean Absolute Error loss for time series regression tasks.
    
    More robust to outliers compared to MSE loss, penalizing
    absolute differences between predictions and targets.
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initialize MAE loss.
        
        Args:
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.mae_loss = nn.L1Loss(reduction=reduction)
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MAE loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        
        loss = self.mae_loss(predictions, targets)
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class HuberLoss(AbstractLoss):
    """
    Huber loss for time series regression tasks.
    
    Combines the best properties of MSE and MAE losses.
    Less sensitive to outliers than MSE while being smoother than MAE.
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean') -> None:
        """
        Initialize Huber loss.
        
        Args:
            delta: Threshold for switching between MSE and MAE behavior
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.huber_loss = nn.HuberLoss(delta=delta, reduction=reduction)
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Huber loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        
        loss = self.huber_loss(predictions, targets)
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class QuantileLoss(AbstractLoss):
    """
    Quantile loss for probabilistic time series forecasting.
    
    Useful for generating prediction intervals and handling
    uncertainty in time series predictions.
    """
    
    def __init__(self, quantile: float = 0.5, reduction: str = 'mean') -> None:
        """
        Initialize Quantile loss.
        
        Args:
            quantile: Target quantile (between 0 and 1)
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction
        
        if not 0 < quantile < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}")
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Quantile loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        
        errors = targets - predictions
        loss = torch.maximum(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class MAPELoss(AbstractLoss):
    """
    Mean Absolute Percentage Error loss for time series regression.
    
    Useful when the scale of errors should be relative to the magnitude
    of the target values. Handles percentage-based error measurement.
    """
    
    def __init__(self, epsilon: float = 1e-8, reduction: str = 'mean') -> None:
        """
        Initialize MAPE loss.
        
        Args:
            epsilon: Small value to avoid division by zero
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MAPE loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        
        # Avoid division by zero
        targets_safe = torch.where(
            torch.abs(targets) < self.epsilon,
            torch.sign(targets) * self.epsilon,
            targets
        )
        
        loss = torch.abs((targets - predictions) / targets_safe)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class SMAPELoss(AbstractLoss):
    """
    Symmetric Mean Absolute Percentage Error loss.
    
    A symmetric version of MAPE that treats over-prediction and
    under-prediction equally, more suitable for time series forecasting.
    """
    
    def __init__(self, epsilon: float = 1e-8, reduction: str = 'mean') -> None:
        """
        Initialize SMAPE loss.
        
        Args:
            epsilon: Small value to avoid division by zero
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SMAPE loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        
        numerator = torch.abs(targets - predictions)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2 + self.epsilon
        
        loss = numerator / denominator
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class WeightedMSELoss(AbstractLoss):
    """
    Weighted Mean Squared Error loss for time series with sample weights.
    
    Useful when certain time points or samples should have more importance
    in the loss computation, such as recent observations or specific events.
    """
    
    def __init__(self, reduction: str = 'mean') -> None:
        """
        Initialize Weighted MSE loss.
        
        Args:
            reduction: Specifies the reduction to apply to the output.
                      Options: 'mean', 'sum', 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted MSE loss between predictions and targets.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' and optionally 'weights' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key
        """
        predictions = model_out['predictions']
        targets = batch['targets']
        weights = batch.get('weights', torch.ones_like(targets))
        
        squared_errors = (predictions - targets) ** 2
        weighted_errors = squared_errors * weights
        
        if self.reduction == 'mean':
            loss = weighted_errors.sum() / weights.sum()
        elif self.reduction == 'sum':
            loss = weighted_errors.sum()
        else:
            loss = weighted_errors
        
        result = model_out.copy()
        result['loss'] = loss
        return result


class CombinedLoss(AbstractLoss):
    """
    Combined loss that allows mixing multiple loss functions.
    
    Useful for multi-objective optimization or when you want to
    balance different aspects of the prediction quality.
    """
    
    def __init__(
        self, 
        losses: Dict[str, AbstractLoss], 
        weights: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize Combined loss.
        
        Args:
            losses: Dictionary mapping loss names to loss instances
            weights: Optional dictionary mapping loss names to weights.
                    If None, all losses are weighted equally.
        """
        super().__init__()
        self.losses = losses
        self.weights = weights or {name: 1.0 for name in losses.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: w / total_weight for name, w in self.weights.items()}
    
    def forward(
        self, 
        model_out: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss from multiple loss functions.
        
        Args:
            model_out: Dictionary containing 'predictions' tensor
            batch: Dictionary containing 'targets' tensor
            
        Returns:
            Dictionary with original model output plus 'loss' key and individual losses
        """
        total_loss = 0.0
        individual_losses = {}
        
        for name, loss_fn in self.losses.items():
            loss_result = loss_fn.forward(model_out, batch)
            individual_loss = loss_result['loss']
            weight = self.weights[name]
            
            total_loss += weight * individual_loss
            individual_losses[f'{name}_loss'] = individual_loss
        
        result = model_out.copy()
        result['loss'] = total_loss
        result.update(individual_losses)
        return result