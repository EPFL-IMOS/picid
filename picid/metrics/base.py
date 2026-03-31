"""Abstract base classes for metric implementations."""

from abc import ABC, abstractmethod
import numpy as np


class AbstractMetric(ABC):
    """
    Abstract base class for all metrics.

    Parameters
    ----------
    name : str
        Metric name used for reporting.
    """

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self):
        """Reset the metric state."""
        pass

    @abstractmethod
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update the metric with new predictions and targets.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions.
        targets : np.ndarray
            Ground-truth targets.
        """
        pass

    @abstractmethod
    def compute(self) -> float:
        """
        Compute and return the final metric value.

        Returns
        -------
        float
            Final metric value.
        """
        pass


class AbstractFileSystemMetric(AbstractMetric):
    """
    Abstract base class for metrics that depend on files on disk.

    Parameters
    ----------
    name : str
        Metric name used for reporting.
    paths : dict, default=None
        File-system paths required by the metric implementation.
    """

    def __init__(self, name: str, paths: dict = None):
        super().__init__(name)
        self.paths = paths
