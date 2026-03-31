from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvalHook(ABC):
    """Abstract base class for all evaluator lifecycle hooks."""

    def on_update_end(self, batch: Dict[str, Any], evaluator: Any) -> None:
        """
        Called after each batch is processed.

        Parameters
        ----------
        batch : Dict[str, Any]
            Current batch payload.
        evaluator : Any
            Evaluator instance calling the hook.
        """

    @abstractmethod
    def on_compute_end(
        self,
        results: Dict[str, float],
        evaluator: Any,
        mode: str,
        epoch: int,
        step: int,
    ) -> None:
        """
        Called after all metrics are computed.

        Parameters
        ----------
        results : Dict[str, float]
            Final computed metrics.
        evaluator : Any
            Evaluator instance calling the hook.
        mode : str
            Evaluation mode.
        epoch : int
            Current epoch index.
        step : int
            Current step index.
        """
