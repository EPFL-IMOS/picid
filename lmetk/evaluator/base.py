"""Abstract evaluator and sequential evaluator implementation."""

from abc import ABC, abstractmethod

from typing import Optional, Any
from lmetk.evaluator.scaling_wrapper import ScalingWrapper

class AbstractEvaluator(ABC):
    """Abstract class for the evaluator class."""

    def __init__(
        self, 
        scaler: Optional[Any] = None, 
        apply_inverse: bool = False, 
        effective_pred_len: Optional[int] = None,
        model_pred_len: Optional[int] = None,
        task_mode: Optional[Any] = False,
        **kwargs
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        scaler : Optional[Any]
            The scaler object with inverse_transform method.
        apply_inverse : bool
            Whether to apply inverse scaling before computing metrics.
        effective_pred_len : Optional[int]
            The effective prediction length to evaluate.
        model_pred_len : Optional[int]
            The model's prediction length.
        task_mode : Optional[Any]
            Task mode for scaling wrapper.
        kwargs : dict
            Additional keyword arguments to set as instance variables.
        """
        super().__init__()
        self.scaling_wrapper = ScalingWrapper(
            scaler=scaler, 
            apply_inverse_scaling=apply_inverse,
            task_mode=task_mode
        )

        self.effective_pred_len = effective_pred_len
        self.model_pred_len = model_pred_len
        self.task_mode = task_mode

        # Automatically set all kwargs as instance variables
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.model_pred_len is None:
            raise ValueError("Model's pred_len (model_pred_len) must be set.")

        if self.effective_pred_len is not None and self.model_pred_len < self.effective_pred_len:
            raise ValueError(
                f"effective_pred_len ({self.effective_pred_len}) cannot be greater than "
                f"model_pred_len ({self.model_pred_len})."
            )



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



