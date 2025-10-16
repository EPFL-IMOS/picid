"""Concrete metric implementations."""

import numpy as np
from .base import AbstractFileSystemMetric, AbstractMetric
import pandas as pd


# This class calculates the mean percentage error using the requested API.
class MeanPercentageErrorMetric(AbstractMetric):
    """
    Mean Percentage Error metric with batch-wise accumulation.
    Calculates: mean((targets - predictions) / targets) * 100
    """

    def __init__(self):
        super().__init__("mpe")
        self.reset()

    def reset(self):
        """Resets the running sum and count to zero."""
        self.running_sum_of_errors = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Processes a batch of predictions and targets to update the metric's state.

        Args:
            predictions (np.ndarray): The estimated RUL values from the model.
            targets (np.ndarray): The actual RUL values.
        """
        assert (
            predictions.shape == targets.shape
        ), "Shape mismatch between predictions and targets"

        # Handle potential division by zero if any target RUL is 0
        epsilon = 1e-9

        # Calculate the error for the batch (as a ratio, not percentage yet)
        errors = (targets - predictions) / (targets + epsilon)

        # Accumulate the sum of errors and the count of samples
        self.running_sum_of_errors += np.sum(errors)
        self.total_count += predictions.size

    def compute(self) -> float:
        """
        Computes the final mean percentage error over all updated batches.

        Returns:
            float: The final metric value.
        """
        if self.total_count == 0:
            raise ValueError("Metric has not been updated with any data.")

        # Calculate the mean and convert to percentage
        return (self.running_sum_of_errors / self.total_count) * 100
