"""Abstract evaluator and sequential evaluator implementation."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Optional, Any
import numpy as np
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.evaluator.scaling_wrapper import ScalingWrapper


class AbstractEvaluator(ABC):
    """Abstract class for the evaluator class."""

    def __init__(self, scaler: Optional[Any] = None, apply_inverse: bool = False):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        scaler : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling before computing metrics
        """
        super().__init__()
        self.scaling_wrapper = ScalingWrapper(scaler, apply_inverse)

    @abstractmethod
    def update(self, model_out: dict):
        """Update the metrics with the model output."""
        pass

    @abstractmethod
    def compute(self):
        """Compute the metrics."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the metrics."""
        pass



