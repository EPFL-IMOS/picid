from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvalHook(ABC):
    """Abstract base class for all evaluator lifecycle hooks."""

    def on_update_end(self, batch: Dict[str, Any], evaluator: Any) -> None:
        """
        Called after each batch is processed.
        Useful for live streaming or partial accumulation.
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
        Primary entry point for saving and plotting.
        """
