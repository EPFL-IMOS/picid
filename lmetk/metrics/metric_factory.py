"""Factory for creating metric instances."""

from typing import Dict, Type
from .base import AbstractMetric
from lmetk.metrics.metrics import (
    MAEMetric,
    MSEMetric,
    NASAScoreMetric,
    RMSEMetric,
    MAPEMetric,
    MSPEMetric,
    RSEMetric,
    CORRMetric,
    NormalizedMAEMetricRailway,
    NormalizedMSEMetricRailway,
)
from lmetk.metrics.rul_metrics import MeanPercentageErrorMetric, PHMScoreMetric


class MetricFactory:
    """Factory class for creating metric instances."""

    _metrics: Dict[str, Type[AbstractMetric]] = {
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

    @classmethod
    def create_metric(cls, metric_name: str, paths: dict) -> AbstractMetric:
        """Create a metric instance by name."""
        metric_name_lower = metric_name.lower()
        if metric_name_lower not in cls._metrics:
            available = list(cls._metrics.keys())
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")

        kwargs = {}
        if metric_name_lower in ["mae_railway", "mse_railway"]:
            kwargs["paths"] = paths

        return cls._metrics[metric_name_lower](**kwargs)

    @classmethod
    def get_available_metrics(cls) -> list:
        """Get list of available metric names."""
        return list(cls._metrics.keys())
