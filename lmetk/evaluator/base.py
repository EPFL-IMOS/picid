"""Abstract evaluator implementation."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict
import logging

from lmetk.evaluator.scaling_wrapper import (
    MultivariateTimeseriesScalingWrapper,
    ScalingWrapper,
)
from lmetk.transforms.base.multisource import InverseTransformMixin

logger = logging.getLogger(__name__)


class AbstractEvaluator(ABC):
    """
    Abstract base class for all evaluators.

    Provides standardized scaling wrapper initialization but stays out of
    the way of subclass configuration.
    """

    def __init__(
        self,
        paths: Optional[Dict[str, str]] = None,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        task_mode: Optional[Any] = False,
        **kwargs,
    ):
        """
        Initialize the base evaluator components.

        Parameters
        ----------
        paths : Optional[dict]
            Path configuration for outputs.
        inverse_transform : Optional[InverseTransformMixin]
            The scaler object with inverse_transform method.
        apply_inverse_scaling : bool
            Whether to apply inverse scaling before computing metrics.
        task_mode : Optional[Any]
            Task mode (e.g. 'multivariate', 'univariate') used to select
            the scaling wrapper strategy.
        **kwargs : dict
            Subclasses may pass additional configuration. This base class
            no longer raises ValueError on unknown keys to allow for
            flexible composition and easier mocking.
        """
        super().__init__()

        # 1. Scaling Strategy Selection
        # We encapsulate the choice of wrapper here, but subclasses
        # use self.scaling_wrapper without caring which one it is.
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

        # 2. Flexible Configuration
        # Instead of failing on unknown kwargs, we simply ignore them or
        # let subclasses handle them. This is the key to fixing the
        # 'Unexpected keyword argument' errors.
        if kwargs:
            logger.debug(f"Base received extra configuration: {list(kwargs.keys())}")

    def __repr__(self):
        # Improved repr that doesn't crash on complex objects
        simple = (int, float, str, bool, type(None))
        attrs = {k: v for k, v in vars(self).items() if isinstance(v, simple)}
        return f"<{self.__class__.__name__}({attrs})>"

    @abstractmethod
    def update(self, model_out: dict) -> None:
        """Update internal state/metrics with batch output."""
        pass

    @abstractmethod
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """Compute final metrics and return a dictionary of results."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state for a fresh evaluation run."""
        pass
