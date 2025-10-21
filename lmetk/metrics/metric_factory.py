"""Factory for creating metric instances."""

from typing import Dict, Type
from .base import AbstractMetric
from lmetk.metrics.metrics import (
    MAEMetric,
    MSEMetric,
    MulticlassAccuracyMetric,
    MulticlassPrecisionMetric,
    MulticlassRecallMetric,
    NASAScoreMetric,
    RMSEMetric,
    MAPEMetric,
    MSPEMetric,
    RSEMetric,
    CORRMetric,
    NormalizedMAEMetricRailway,
    NormalizedMSEMetricRailway,
    MulticlassF1Metric,
)
from lmetk.metrics.rul_metrics import MeanPercentageErrorMetric, PHMScoreMetric


class MetricFactory:
    """Factory class for creating metric instances."""

    _regression_metrics: Dict[str, Type[AbstractMetric]] = {
        "mae": MAEMetric,
        "mse": MSEMetric,
        "rmse": RMSEMetric,
        "mape": MAPEMetric,
        "mspe": MSPEMetric,
        "rse": RSEMetric,
        "corr": CORRMetric,
        "mae_railway": NormalizedMAEMetricRailway,
        "mse_railway": NormalizedMSEMetricRailway,
        "nasa_score": NASAScoreMetric,
        "mpe": MeanPercentageErrorMetric,
        "phm_score": PHMScoreMetric,
    }

    _classification_metrics: Dict[str, Type[AbstractMetric]] = {
        "accuracy": MulticlassAccuracyMetric,
        "precision": MulticlassPrecisionMetric,
        "recall": MulticlassRecallMetric,
        "f1": MulticlassF1Metric,
    }

    @classmethod
    def create_metric(cls, metric_name: str, paths: dict) -> AbstractMetric:
        """Create a metric instance by name."""
        metric_name_lower = metric_name.lower()
        if metric_name_lower not in cls._regression_metrics:
            available = list(cls._regression_metrics.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")

        kwargs = {}
        if metric_name_lower in ["mae_railway", "mse_railway"]:
            kwargs["paths"] = paths

        return cls._regression_metrics[metric_name_lower](**kwargs)

    @classmethod
    def create_classification_metric(
        cls, metric_name: str, num_classes: int
    ) -> AbstractMetric:
        """Create a classification metric instance by name."""
        metric_name_lower = metric_name.lower()
        if metric_name_lower not in cls._classification_metrics:
            available = list(cls._classification_metrics.keys())
            raise ValueError(
                f"Unknown classification metric '{metric_name}'. Available: {available}"
            )

        return cls._classification_metrics[metric_name_lower](num_classes=num_classes)

    @classmethod
    def get_available_metrics(cls) -> list:
        """Get list of available metric names."""
        return list(cls._regression_metrics.keys())
