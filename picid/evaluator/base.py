from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import logging

from picid.evaluator.scaling_wrapper import (
    MultivariateTimeseriesScalingWrapper,
    ScalingWrapper,
)
from picid.transforms.base.multisource import InverseTransformMixin

logger = logging.getLogger(__name__)


class AbstractEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    Handles scaling strategy selection and standardizes the interface.
    """

    def __init__(
        self,
        paths: Optional[Dict[str, str]] = None,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()

        # Strategy selection
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

    def __repr__(self):
        simple = (int, float, str, bool, type(None))
        attrs = {k: v for k, v in vars(self).items() if isinstance(v, simple)}
        return f"<{self.__class__.__name__}({attrs})>"

    @abstractmethod
    def update(self, model_out: dict) -> None:
        """Update internal state/metrics with batch output."""
        pass

    @abstractmethod
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """Compute final metrics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
        pass
