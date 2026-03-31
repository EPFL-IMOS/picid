"""Factory for creating metric instances."""

from typing import Dict, Type
from .base import AbstractMetric
from picid.metrics.metrics import (
    MAEMetric,
    MSEMetric,
    MulticlassAUROCMetric,
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
from picid.metrics.rul_metrics import MeanPercentageErrorMetric, PHMScoreMetric


class MetricFactory:
    """
    Factory class for creating metric instances.

    The factory keeps the metric selection logic in one place so experiment
    code can request metrics by name without knowing the concrete classes.
    """

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
        "auroc": MulticlassAUROCMetric,
    }

    @classmethod
    def create_metric(cls, metric_name: str, paths: dict) -> AbstractMetric:
        """
        Create a regression metric instance by name.

        Parameters
        ----------
        metric_name : str
            Name of the metric to create.
        paths : dict
            File-system paths required by filesystem-backed metrics.

        Returns
        -------
        AbstractMetric
            Instantiated metric implementation.
        """
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
        """
        Create a classification metric instance by name.

        Parameters
        ----------
        metric_name : str
            Name of the metric to create.
        num_classes : int
            Number of classes for the metric.

        Returns
        -------
        AbstractMetric
            Instantiated classification metric.
        """
        metric_name_lower = metric_name.lower()
        if metric_name_lower not in cls._classification_metrics:
            available = list(cls._classification_metrics.keys())
            raise ValueError(
                f"Unknown classification metric '{metric_name}'. Available: {available}"
            )

        return cls._classification_metrics[metric_name_lower](num_classes=num_classes)

    @classmethod
    def get_available_metrics(cls) -> list:
        """
        Get the list of available regression metric names.

        Returns
        -------
        list
            Metric names registered in the regression metric registry.
        """
        return list(cls._regression_metrics.keys())
