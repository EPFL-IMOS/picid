"""Abstract class for the optimizer manager class."""

from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):
    """Abstract class for optimizer manager implementations."""

    @abstractmethod
    def configure_optimizer(self, model_parameters: dict):
        """
        Configure the optimizer and scheduler.

        Act as a wrapper.

        Parameters
        ----------
        model_parameters : dict
            The model parameters.
        """
