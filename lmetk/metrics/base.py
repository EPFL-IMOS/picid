"""Abstract base class for metrics."""

from abc import ABC, abstractmethod
import numpy as np


class AbstractMetric(ABC):
    """Abstract base class for all metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    @abstractmethod
    def reset(self):
        """Reset the metric state."""
        pass
    
    @abstractmethod
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Update the metric with new predictions and targets."""
        pass
    
    @abstractmethod
    def compute(self) -> float:
        """Compute and return the final metric value."""
        pass
