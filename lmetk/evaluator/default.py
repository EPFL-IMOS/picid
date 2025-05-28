"""
Transformer-specific evaluator implementation for time series forecasting.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from .base import AbstractEvaluator


class DefaultEvaluator(AbstractEvaluator):
    """
    Evaluator for transformer-based time series forecasting models.
    
    This evaluator computes standard forecasting metrics including MAE, MSE, 
    RMSE, MAPE, and MSPE. It accumulates predictions and ground truth values
    across batches and computes final metrics.
    
    Attributes:
        predictions (List[np.ndarray]): Accumulated model predictions
        ground_truths (List[np.ndarray]): Accumulated ground truth values
        losses (List[float]): Accumulated loss values
        inverse_transform (bool): Whether to apply inverse transform to predictions
        scaler (Optional[Any]): Data scaler for inverse transformation
    """
    
    def __init__(self, inverse_transform: bool = False, scaler: Optional[Any] = None):
        """
        Initialize the transformer evaluator.
        
        Args:
            inverse_transform: Whether to apply inverse transform to predictions
            scaler: Data scaler for inverse transformation (if needed)
        """
        super().__init__()
        self.inverse_transform = inverse_transform
        self.scaler = scaler
        self.reset()
    
    def update(self, model_out: dict) -> None:
        """
        Update the metrics with the model output.
        
        Expected model_out format:
        {
            'predictions': torch.Tensor or np.ndarray,
            'ground_truth': torch.Tensor or np.ndarray,
            'loss': float (optional)
        }
        
        Args:
            model_out: Dictionary containing predictions, ground truth, and optionally loss
            
        Raises:
            KeyError: If required keys are missing from model_out
            ValueError: If tensor shapes don't match
        """
        if 'predictions' not in model_out or 'ground_truth' not in model_out:
            raise KeyError("model_out must contain 'predictions' and 'ground_truth' keys")
        
        predictions = model_out['predictions']
        ground_truth = model_out['ground_truth']
        
        # Convert to numpy if tensors
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        
        # Validate shapes
        if predictions.shape != ground_truth.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs ground_truth {ground_truth.shape}")
        
        # Apply inverse transform if specified
        if self.inverse_transform and self.scaler is not None:
            predictions = self._apply_inverse_transform(predictions)
            ground_truth = self._apply_inverse_transform(ground_truth)
        
        # Store predictions and ground truth
        self.predictions.append(predictions)
        self.ground_truths.append(ground_truth)
        
        # Store loss if provided
        if 'loss' in model_out:
            self.losses.append(float(model_out['loss']))
    
    def _apply_inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply inverse transformation to data.
        
        Args:
            data: Data to inverse transform
            
        Returns:
            Inverse transformed data
        """
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        # Reshape for scaler (assumes scaler expects 2D input)
        if len(original_shape) > 2:
            data_reshaped = data.reshape(-1, original_shape[-1])
            transformed = self.scaler.inverse_transform(data_reshaped)
            return transformed.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute the evaluation metrics.
        
        Returns:
            Dictionary containing computed metrics:
            - mae: Mean Absolute Error
            - mse: Mean Squared Error  
            - rmse: Root Mean Squared Error
            - mape: Mean Absolute Percentage Error
            - mspe: Mean Squared Percentage Error
            - avg_loss: Average loss (if losses were provided)
            
        Raises:
            ValueError: If no data has been accumulated
        """
        if len(self.predictions) == 0:
            raise ValueError("No predictions available. Call update() first.")
        
        # Concatenate all predictions and ground truths
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_ground_truths = np.concatenate(self.ground_truths, axis=0)
        
        # Flatten for metric computation
        pred_flat = all_predictions.reshape(-1)
        true_flat = all_ground_truths.reshape(-1)
        
        # Compute metrics
        metrics = self._compute_metrics(pred_flat, true_flat)
        
        # Add average loss if available
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        return metrics
    
    def _compute_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute forecasting metrics.
        
        Args:
            predictions: Flattened predictions array
            ground_truth: Flattened ground truth array
            
        Returns:
            Dictionary of computed metrics
        """
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - ground_truth))
        
        # Mean Squared Error
        mse = np.mean((predictions - ground_truth) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Percentage Error (avoid division by zero)
        epsilon = 1e-8
        mape = np.mean(np.abs((ground_truth - predictions) / (ground_truth + epsilon))) * 100
        
        # Mean Squared Percentage Error
        mspe = np.mean(((ground_truth - predictions) / (ground_truth + epsilon)) ** 2) * 100
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'mspe': float(mspe)
        }
    
    def reset(self) -> None:
        """Reset the evaluator state."""
        self.predictions: List[np.ndarray] = []
        self.ground_truths: List[np.ndarray] = []
        self.losses: List[float] = []
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics including per-sample statistics.
        
        Returns:
            Dictionary with detailed metrics and statistics
        """
        if len(self.predictions) == 0:
            raise ValueError("No predictions available. Call update() first.")
        
        # Get basic metrics
        basic_metrics = self.compute()
        
        # Concatenate all data
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_ground_truths = np.concatenate(self.ground_truths, axis=0)
        
        # Compute per-sample errors
        absolute_errors = np.abs(all_predictions - all_ground_truths)
        squared_errors = (all_predictions - all_ground_truths) ** 2
        
        detailed_metrics = {
            **basic_metrics,
            'num_samples': len(all_predictions.reshape(-1)),
            'prediction_std': float(np.std(all_predictions)),
            'ground_truth_std': float(np.std(all_ground_truths)),
            'max_absolute_error': float(np.max(absolute_errors)),
            'min_absolute_error': float(np.min(absolute_errors)),
            'median_absolute_error': float(np.median(absolute_errors)),
            'q95_absolute_error': float(np.percentile(absolute_errors, 95)),
            'q05_absolute_error': float(np.percentile(absolute_errors, 5)),
        }
        
        return detailed_metrics
    
    def get_predictions_and_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get accumulated predictions and ground truth values.
        
        Returns:
            Tuple of (predictions, ground_truths) as numpy arrays
        """
        if len(self.predictions) == 0:
            return np.array([]), np.array([])
        
        predictions = np.concatenate(self.predictions, axis=0)
        ground_truths = np.concatenate(self.ground_truths, axis=0)
        
        return predictions, ground_truths
    
    def save_predictions(self, filepath: str) -> None:
        """
        Save predictions and ground truth to a numpy file.
        
        Args:
            filepath: Path to save the predictions (without extension)
        """
        if len(self.predictions) == 0:
            raise ValueError("No predictions to save. Call update() first.")
        
        predictions, ground_truths = self.get_predictions_and_targets()
        
        np.save(f"{filepath}_predictions.npy", predictions)
        np.save(f"{filepath}_ground_truth.npy", ground_truths)
    
    def __len__(self) -> int:
        """Return the number of accumulated samples."""
        return len(self.predictions)
    
    def __str__(self) -> str:
        """String representation of the evaluator."""
        num_samples = len(self.predictions)
        return f"TransformerEvaluator(samples={num_samples}, inverse_transform={self.inverse_transform})"