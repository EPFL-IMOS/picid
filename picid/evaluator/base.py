# numpydoc ignore=GL08
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

    Parameters
    ----------
    paths : dict[str, str] | None, default=None
        File-system paths required by some evaluators.
    inverse_transform : InverseTransformMixin | None, default=None
        Optional inverse transform used for scaling-aware evaluation.
    apply_inverse_scaling : bool, default=False
        Whether inverse scaling should be applied before metric computation.
    task_mode : Any | None, default=None
        Task mode used to select the scaling wrapper implementation.
        Use ``"multivariate"`` for multi-channel series (e.g. UMAR forecasting)
        so inverse scaling uses :class:`~picid.evaluator.scaling_wrapper.MultivariateTimeseriesScalingWrapper`.
    **kwargs
        Additional evaluator-specific keyword arguments.
    """

    def __init__(  # numpydoc ignore=GL08
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
        """
        Update internal state and metrics with batch output.

        Parameters
        ----------
        model_out : dict
            Model output dictionary containing predictions and targets.
        """
        pass

    @abstractmethod
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Compute final metrics.

        Parameters
        ----------
        mode : str
            Evaluation mode.
        epoch : int
            Current epoch index.
        step : int
            Current step index.

        Returns
        -------
        Dict[str, float]
            Computed metrics.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state."""
        pass
