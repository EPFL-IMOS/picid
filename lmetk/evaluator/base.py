"""Abstract evaluator and sequential evaluator implementation."""

from abc import ABC, abstractmethod

from typing import Optional, Any
from lmetk.evaluator.scaling_wrapper import ScalingWrapper

class AbstractEvaluator(ABC):
    """Abstract class for the evaluator class."""

    def __init__(self, scaler: Optional[Any] = None, apply_inverse: bool = False, **kwargs):
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
        self.scaling_wrapper = ScalingWrapper(
            scaler=scaler, 
            apply_inverse_scaling=apply_inverse,
            task_mode=kwargs.get("task_mode", False)
            )

        
        # Automatically set all kwargs as instance variables
        for key, value in kwargs.items():
            setattr(self, key, value)


    def __repr__(self):
        cls_name = self.__class__.__name__
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"<{cls_name}({attrs})>"

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



