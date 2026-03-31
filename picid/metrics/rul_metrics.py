"""Concrete metric implementations."""

import numpy as np
from .base import AbstractMetric


# This class calculates the mean percentage error using the requested API.
class MeanPercentageErrorMetric(AbstractMetric):
    """
    Mean percentage error metric with batch-wise accumulation.

    The metric computes ``mean((targets - predictions) / targets) * 100``
    over all accumulated batches.
    """

    def __init__(self):
        super().__init__("mpe")
        self.reset()

    def reset(self):
        """Reset the running sum and count to zero."""
        self.running_sum_of_errors = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update the metric state with a batch of predictions and targets.

        Parameters
        ----------
        predictions : np.ndarray
            The estimated RUL values from the model.
        targets : np.ndarray
            The actual RUL values.
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
        Compute the final mean percentage error over all updated batches.

        Returns
        -------
        float
            The final metric value.
        """
        if self.total_count == 0:
            raise ValueError("Metric has not been updated with any data.")

        # Calculate the mean and convert to percentage
        return (self.running_sum_of_errors / self.total_count) * 100


class PHMScoreMetric(AbstractMetric):
    """
    Calculate the scoring function from the IEEE 2012 PHM Data Challenge.

    The score is an average of an asymmetric function ``A_i`` that depends on
    the percentage error of the RUL prediction.

    Ref: H. Dhungana et al., "Bearing Prognostics Using the PRONOSTIA Data:
    A Comparative Study" (Eq. 1, 2, and 3).
    """

    def __init__(self):
        super().__init__("phm_score")
        self.reset()
        # Pre-calculate ln(0.5) for efficiency
        self._ln_half = np.log(0.5)

    def reset(self):
        """Reset the running sum of scores and the total count."""
        self.total_sum_of_A = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update the metric state with a batch of predictions and targets.

        Parameters
        ----------
        predictions : np.ndarray
            Model predictions.
        targets : np.ndarray
            Ground-truth targets.
        """
        assert (
            predictions.shape == targets.shape
        ), "Shape mismatch between predictions and targets"

        epsilon = 1e-9
        # Step 1: Calculate the percentage error for each sample in the batch
        percent_errors = 100 * (targets - predictions) / (targets + epsilon)

        # Step 2: Calculate A_i for each sample using the asymmetric formula
        late_prediction_scores = np.exp(-self._ln_half * (percent_errors / 5.0))
        early_prediction_scores = np.exp(self._ln_half * (percent_errors / 20.0))

        A_i_batch = np.where(
            percent_errors <= 0, late_prediction_scores, early_prediction_scores
        )

        # Step 3: Accumulate the sum of A_i scores and the sample count
        self.total_sum_of_A += np.sum(A_i_batch)
        self.total_count += predictions.size

    def compute(self) -> float:
        """
        Compute the final mean score over all updated batches.

        Returns
        -------
        float
            The final metric value.
        """
        if self.total_count == 0:
            raise ValueError("Metric has not been updated with any data.")
        return self.total_sum_of_A / self.total_count
