"""Abstract evaluator and sequential evaluator implementation."""

from abc import ABC, abstractmethod

from typing import Optional, Any
from lmetk.evaluator.scaling_wrapper import (
    MultivariateTimeseriesScalingWrapper,
    ScalingWrapper,
)
from lmetk.transforms.base.multisource import InverseTransformMixin


class AbstractEvaluator(ABC):
    """Abstract class for the evaluator class."""

    def __init__(
        self,
        paths: dict[str, str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: Optional[Any] = False,
        **kwargs,
    ):
        """
        Initialize the evaluator.

        Parameters
        ----------
        inverse_transform : Optional[InverseTransformMixin]
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

        if task_mode == "multivariate":
            self.scaling_wrapper = MultivariateTimeseriesScalingWrapper(
                inverse_transform=inverse_transform,
                apply_inverse_scaling=apply_inverse_scaling,
                task_mode=task_mode,
            )
        else:
            self.scaling_wrapper = ScalingWrapper(
                inverse_transform=inverse_transform,
                apply_inverse_scaling=apply_inverse_scaling,
                task_mode=task_mode,
            )

        self.task_mode = task_mode
        self.paths = paths

        # Check if kwargs is empty and raise exception if not empty
        for k, v in kwargs.items():
            if k not in ["inverse_transform_name"]:
                raise ValueError(f"Unexpected keyword argument: {k}")

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
