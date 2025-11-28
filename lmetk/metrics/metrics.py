"""Concrete metric implementations."""

import numpy as np
import pandas as pd
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from .base import AbstractFileSystemMetric, AbstractMetric


class NormalizedMAEMetricRailway(AbstractFileSystemMetric):
    """Normalized Mean Absolute Error metric for Railvay."""

    def __init__(self, paths: dict = None):
        super().__init__("mae", paths)
        self.factor = (
            pd.read_csv(f"{paths['data_dir']}/railway/railway.csv")["Bahnlast"]
            .abs()
            .mean()
        )

    def reset(self):
        self.values = []

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        abs_errors = (np.mean(np.abs(predictions - targets)) * 100) / self.factor
        self.values.extend(abs_errors.flatten())

    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class NormalizedMSEMetricRailway(AbstractFileSystemMetric):
    """ "Normalized Mean Squared Error metric for Railvay."""

    def __init__(self, paths: dict = None):
        super().__init__("mse", paths)
        self.factor = (
            pd.read_csv(f"{paths['data_dir']}/railway/railway.csv")["Bahnlast"]
            .abs()
            .mean()
        )

    def reset(self):
        self.values = []

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        squared_errors = (np.mean((predictions - targets) ** 2) * 100) / self.factor
        self.values.extend(squared_errors.flatten())

    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class MAEMetric(AbstractMetric):
    """Mean Absolute Error metric with proper batch accumulation."""

    def __init__(self):
        super().__init__("mae")
        self.reset()

    def reset(self):
        """Resets the accumulators to zero."""
        self.total_absolute_error = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Updates the metric's state with a new batch of predictions and targets.
        """
        # 1. Calculate the absolute errors for every element in the batch
        abs_errors = np.abs(predictions - targets)

        # 2. Accumulate the sum of errors and the total element count
        self.total_absolute_error += np.sum(abs_errors)
        self.total_count += predictions.size

    def compute(self) -> float:
        """
        Computes the final MAE from the accumulated values.
        """
        if self.total_count == 0:
            raise ValueError("No data available for computation")

        # 3. Perform the final division only once
        return self.total_absolute_error / self.total_count


class MSEMetric(AbstractMetric):
    """Mean Squared Error metric with batch-wise accumulation for (B, L, 1) shapes."""

    def __init__(self):
        super().__init__("mse")
        self.reset()

    def reset(self):
        self.total_squared_error = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Ensure shapes are compatible
        assert (
            predictions.shape == targets.shape
        ), "Shape mismatch between predictions and targets"

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Accumulate
        self.total_squared_error += np.sum(squared_errors)
        self.total_count += predictions.size

    def compute(self) -> float:
        if self.total_count == 0:
            raise ValueError("No data available for computation")
        return self.total_squared_error / self.total_count


class RMSEMetric(AbstractMetric):
    """Root Mean Squared Error metric with batch-wise accumulation for (B, L, 1) shapes."""

    def __init__(self):
        super().__init__("rmse")
        self.reset()

    def reset(self):
        self.total_squared_error = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Ensure shapes match
        assert (
            predictions.shape == targets.shape
        ), "Shape mismatch between predictions and targets"

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Accumulate sum and count
        self.total_squared_error += np.sum(squared_errors)
        self.total_count += np.prod(targets.shape)  # B * L * 1

    def compute(self) -> float:
        if self.total_count == 0:
            raise ValueError("No data available for computation")
        return np.sqrt(self.total_squared_error / self.total_count)


class MAPEMetric(AbstractMetric):
    """Mean Absolute Percentage Error metric."""

    def __init__(self):
        super().__init__("mape")

    def reset(self):
        self.values = []

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Avoid division by zero
        mask = targets != 0
        if np.any(mask):
            percentage_errors = np.abs(
                (predictions[mask] - targets[mask]) / targets[mask]
            )
            self.values.extend(percentage_errors.flatten())

    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class MSPEMetric(AbstractMetric):
    """Mean Squared Percentage Error metric."""

    def __init__(self):
        super().__init__("mspe")

    def reset(self):
        self.values = []

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        # Avoid division by zero
        mask = targets != 0
        if np.any(mask):
            percentage_errors = (
                (predictions[mask] - targets[mask]) / targets[mask]
            ) ** 2
            self.values.extend(percentage_errors.flatten())

    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class RSEMetric(AbstractMetric):
    """Relative Squared Error metric."""

    def __init__(self):
        super().__init__("rse")

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())

    def compute(self) -> float:
        if not self.predictions:
            raise ValueError("No data available for computation")

        pred = np.array(self.predictions)
        true = np.array(self.targets)

        numerator = np.sqrt(np.sum((true - pred) ** 2))
        denominator = np.sqrt(np.sum((true - true.mean()) ** 2))

        return numerator / denominator


class CORRMetric(AbstractMetric):
    """
    Correlation coefficient metric using a memory-efficient online algorithm.
    """

    def __init__(self):
        super().__init__("corr")
        self.reset()

    def reset(self):
        """Initializes the accumulators for the online algorithm."""
        self.n = 0
        self.sum_x = 0.0  # Sum of targets
        self.sum_y = 0.0  # Sum of predictions
        self.sum_x_sq = 0.0  # Sum of targets squared
        self.sum_y_sq = 0.0  # Sum of predictions squared
        self.sum_xy = 0.0  # Sum of the product of targets and predictions

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Updates the metric's state with a new batch."""
        x = targets.flatten()
        y = predictions.flatten()

        # Update accumulators with the data from the current batch
        self.n += x.size
        self.sum_x += np.sum(x)
        self.sum_y += np.sum(y)
        self.sum_x_sq += np.sum(x**2)
        self.sum_y_sq += np.sum(y**2)
        self.sum_xy += np.sum(x * y)

    def compute(self) -> float:
        """Computes the final correlation coefficient from the accumulated values."""
        if self.n == 0:
            raise ValueError("No data available for computation")

        # Numerator of the Pearson correlation formula
        numerator = self.n * self.sum_xy - self.sum_x * self.sum_y

        # Denominator of the Pearson correlation formula
        var_x = self.n * self.sum_x_sq - self.sum_x**2
        var_y = self.n * self.sum_y_sq - self.sum_y**2

        # Avoid division by zero if one of the variables has zero variance
        if var_x <= 0 or var_y <= 0:
            return 0.0

        denominator = np.sqrt(var_x * var_y)

        return numerator / denominator


class NASAScoreMetric(AbstractMetric):
    """NASA score metric."""

    def __init__(self):
        super().__init__("nasa_score")

    def reset(self):
        self.values = []

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        diff = predictions - targets
        score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
        self.values.extend(score.flatten())

    def compute(self) -> float:
        if not self.values:
            raise ValueError("No data available for computation")
        return np.mean(self.values)


class MASEMetric(AbstractMetric):
    """Mean Absolute Scaled Error (MASE) metric supporting multi-feature inputs.

    Reference:
        Hyndman, R. J., & Koehler, A. B. (2006).
        Another look at measures of forecast accuracy.
        International Journal of Forecasting, 22(4), 679–688.
    """

    def __init__(self, training_series: np.ndarray = None, seasonality: int = 1):
        """
        Args:
            training_series: Array of historical values used for scaling (T, F) or (T,) shape.
            seasonality: Periodicity used for the naive forecast (default=1 for non-seasonal data).
        """
        super().__init__("mase")
        self.training_series = training_series
        self.seasonality = seasonality
        self.reset()

        if training_series is not None:
            self.scale = self._compute_scale(training_series, seasonality)
        else:
            self.scale = None

    def _compute_scale(self, series: np.ndarray, m: int) -> float:
        """Computes scaling factor as mean absolute naive forecast error."""
        series = np.asarray(series)
        if series.ndim == 1:
            diffs = np.abs(series[m:] - series[:-m])
        else:
            diffs = np.abs(series[m:, :] - series[:-m, :]).flatten()
        scale = np.mean(diffs)
        if scale == 0:
            raise ValueError(
                "Zero scaling factor in MASE; training series may be constant."
            )
        return scale

    def reset(self):
        """Resets accumulated errors."""
        self.total_absolute_error = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """Accumulates absolute errors for batched data."""
        if self.scale is None:
            raise ValueError(
                "Scale not initialized. Provide a training_series during initialization."
            )

        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}"
            )

        abs_errors = np.abs(predictions - targets)
        self.total_absolute_error += np.sum(abs_errors)
        self.total_count += np.prod(targets.shape)

    def compute(self) -> float:
        """Computes final Mean Absolute Scaled Error."""
        if self.total_count == 0:
            raise ValueError("No data available for computation")

        mase = (self.total_absolute_error / self.total_count) / self.scale
        return mase


class MulticlassAccuracyMetric(AbstractMetric):
    def __init__(self, num_classes: int):
        self.metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
        super().__init__("accuracy")

    def reset(self):
        self.metric.reset()

    def update(self, predictions, targets):
        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        preds = (
            torch.argmax(predictions, dim=-1, keepdim=True) if predictions.shape[-1] > 1 else predictions
        )
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute().item()


class MulticlassPrecisionMetric(AbstractMetric):
    def __init__(self, num_classes: int):
        self.metric = MulticlassPrecision(num_classes=num_classes, average="macro")
        super().__init__("precision")

    def reset(self):
        self.metric.reset()

    def update(self, predictions, targets):
        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        preds = (
            torch.argmax(predictions, dim=-1, keepdim=True) if predictions.shape[-1] > 1 else predictions
        )
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute().item()


class MulticlassRecallMetric(AbstractMetric):
    def __init__(self, num_classes: int):
        self.metric = MulticlassRecall(num_classes=num_classes, average="macro")
        super().__init__("recall")

    def reset(self):
        self.metric.reset()

    def update(self, predictions, targets):
        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        preds = (
            torch.argmax(predictions, dim=-1, keepdim=True) if predictions.shape[-1] > 1 else predictions
        )
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute().item()


class MulticlassF1Metric(AbstractMetric):
    def __init__(self, num_classes: int):
        self.metric = MulticlassF1Score(num_classes=num_classes, average="macro")
        super().__init__("f1")

    def reset(self):
        self.metric.reset()

    def update(self, predictions, targets):
        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        preds = (
            torch.argmax(predictions, dim=-1, keepdim=True) if predictions.shape[-1] > 1 else predictions
        )
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute().item()

class MulticlassAUROCMetric(AbstractMetric):
    def __init__(self, num_classes: int):
        self.metric = MulticlassAUROCMetric(num_classes=num_classes, average="macro")
        super().__init__("AUROC")

    def reset(self):
        self.metric.reset()

    def update(self, predictions, targets):
        predictions = torch.from_numpy(predictions)
        targets = torch.from_numpy(targets)

        preds = (
            torch.argmax(predictions, dim=-1, keepdim=True) if predictions.shape[-1] > 1 else predictions
        )
        self.metric.update(preds, targets)

    def compute(self):
        return self.metric.compute().item()