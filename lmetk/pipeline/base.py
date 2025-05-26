"""
Base pipeline strategy for transformer model training and evaluation.

This module provides an abstract base class for implementing different
transformer training strategies with standardized interfaces for setup,
training, evaluation, and model persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

from einops import rearrange
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tqdm import tqdm



# Suppress flash attention warning
warnings.filterwarnings(
    "ignore", 
    message=".*Torch was not compiled with flash attention.*"
)


class BasePipelineStrategy(ABC):
    """
    Abstract base class for transformer training pipeline strategies.
    
    This class defines the standard interface for implementing different
    training strategies for transformer models. Subclasses must implement
    all abstract methods to provide specific functionality.
    
    Attributes:
        cfg (DictConfig): Configuration object containing experiment parameters
        datamodule (Any): PyTorch Lightning datamodule for data loading
        model (Any): PyTorch model to be trained
        evaluator (Any): Evaluator object for model assessment
        logger (Any): Logger for experiment tracking
        optimizer (Any): Optimizer for model training
        loss (Any): Loss function for training
        _is_setup (bool): Flag indicating if setup has been completed
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        datamodule: Any,
        model: Any,
        evaluator: Any,
        logger: Any,
        optimizer: Any,
        loss: Any
    ) -> None:
        """
        Initialize the pipeline strategy.
        
        Args:
            cfg: Configuration object with experiment parameters
            datamodule: PyTorch Lightning datamodule for data loading
            model: PyTorch model to be trained
            evaluator: Evaluator object for model assessment
            logger: Logger for experiment tracking
            optimizer: Optimizer for model training
            loss: Loss function for training
            
        Raises:
            ValueError: If required configuration parameters are missing
        """
        self._validate_config(cfg)
        
        self.cfg = cfg
        self.datamodule = datamodule
        self.model = model
        self.evaluator = evaluator
        self.logger = logger
        self.optimizer = optimizer
        self.loss = loss
        self._is_setup = False
    
    def _validate_config(self, cfg: DictConfig) -> None:
        """
        Validate the configuration object.
        
        Args:
            cfg: Configuration to validate
            
        Raises:
            ValueError: If required configuration parameters are missing
        """
        if not hasattr(cfg, 'info') or not hasattr(cfg.info, 'exp_name'):
            raise ValueError("Configuration must contain 'info.exp_name'")

    
    @abstractmethod
    def setup(self) -> None:
        """
        Setup the pipeline strategy.
        
        This method should initialize all necessary components for training,
        including data loaders, model initialization, optimizer setup, etc.
        
        Must be called before run() or evaluate().
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute the training phase.
        
        Returns:
            Dictionary containing training results and metrics
            
        Raises:
            RuntimeError: If setup() has not been called
        """
        pass
    
    @abstractmethod
    def test(self) -> Dict[str, Any]:
        """
        Execute the testing phase.
        
        Returns:
            Dictionary containing test results and metrics
            
        Raises:
            RuntimeError: If setup() has not been called
        """
        pass
    
    @abstractmethod
    def predict(self, data_loader: Optional[Any] = None) -> Any:
        """
        Generate predictions using the trained model.
        
        Args:
            data_loader: Optional data loader for prediction. If None,
                        uses the default prediction data loader.
        
        Returns:
            Model predictions
            
        Raises:
            RuntimeError: If setup() has not been called
        """
        pass
    
    @abstractmethod
    def evaluate(self, data_loader: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            data_loader: Optional data loader for evaluation. If None,
                        uses the default evaluation data loader.
        
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            RuntimeError: If setup() has not been called
        """
        pass
    
    @abstractmethod
    def save(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Save model checkpoint and training state.
        
        Args:
            checkpoint_path: Path where to save the checkpoint
            
        Raises:
            IOError: If unable to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model checkpoint and training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            IOError: If unable to load checkpoint
        """
        pass

    
    def _ensure_setup(self) -> None:
        """
        Ensure that setup has been called before executing other methods.
        
        Raises:
            RuntimeError: If setup() has not been called
        """
        if not self._is_setup:
            raise RuntimeError(
                "setup() must be called before running the pipeline. "
                "Call setup() first."
            )
    
    @property
    def experiment_name(self) -> str:
        """Get the experiment name from configuration."""
        return self.cfg.info.exp_name
    
    @property
    def is_setup(self) -> bool:
        """Check if the pipeline has been set up."""
        return self._is_setup
    
    def __str__(self) -> str:
        """String representation of the pipeline strategy."""
        status = "configured" if self._is_setup else "not configured"
        return f"{self.__class__.__name__}(experiment='{self.experiment_name}', status='{status}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the pipeline strategy."""
        return (
            f"{self.__class__.__name__}("
            f"experiment='{self.experiment_name}', "
            f"setup={self._is_setup}, "
            f"lags={self.lags})"
        )