# numpydoc ignore=GL08
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from picid.metrics.metric_factory import MetricFactory

logger = logging.getLogger(__name__)


class MetricManager:
    """
    Orchestrate multiple metric instances.

    Handles dual-scaling (normalized/denormalized) and task-specific
    aggregation.

    Parameters
    ----------
    metric_names : List[str]
        Metric names to manage.
    task_type : str
        Task type: ``"classification"`` or a regression-style task (``"regression"``,
        ``"rul"``, ``"forecasting"``, etc.). Forecasting is aggregated like regression.
    paths : Optional[Dict[str, str]], default=None
        File-system paths required by filesystem-backed metrics.
    num_classes : Optional[int], default=None
        Number of classes for classification metrics.
    is_dual : bool, default=False
        Whether to track both raw and normalized metrics.
    """

    def __init__(
        self,
        metric_names: List[str],
        task_type: str,
        paths: Optional[Dict[str, str]] = None,
        num_classes: Optional[int] = None,
        is_dual: bool = False,
    ):
        """
        Initialize the metric manager and build the managed metric sets.

        Parameters
        ----------
        metric_names : List[str]
            Metric names to manage.
        task_type : str
            Task type: ``"classification"`` or regression-style (including ``"forecasting"``).
        paths : Optional[Dict[str, str]], default=None
            File-system paths required by filesystem-backed metrics.
        num_classes : Optional[int], default=None
            Number of classes for classification metrics.
        is_dual : bool, default=False
            Whether to track both raw and normalized metrics.
        """
        if not metric_names or len(metric_names) == 0:
            raise ValueError("metric_names cannot be empty")

        self.metric_names = [n.lower() for n in metric_names]
        self.task_type = task_type.lower()
        self.paths = paths
        self.num_classes = num_classes
        self.is_dual = is_dual

        # Initialize internal metric containers
        self.metrics = self._create_metric_set()
        self.normalized_metrics = self._create_metric_set() if is_dual else None

    def _create_metric_set(self) -> Dict[str, Any]:
        """
        Use :class:`MetricFactory` to build the required metric objects.

        Returns
        -------
        Dict[str, Any]
            Mapping of metric names to instantiated metric objects.
        """
        res = {}
        for name in self.metric_names:
            if self.task_type == "classification":
                if self.num_classes is None:
                    raise ValueError(
                        f"Metric '{name}' requires num_classes for classification."
                    )
                res[name] = MetricFactory.create_classification_metric(
                    name, self.num_classes
                )
            else:
                # Regression, railway, forecasting, etc.
                res[name] = MetricFactory.create_metric(name, self.paths or {})
        return res

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        norm_preds: Optional[np.ndarray] = None,
        norm_targets: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update all managed metrics with a new batch of predictions and targets.

        Parameters
        ----------
        predictions : np.ndarray
            Raw model predictions.
        targets : np.ndarray
            Raw ground-truth targets.
        norm_preds : Optional[np.ndarray], default=None
            Normalized predictions used in dual mode.
        norm_targets : Optional[np.ndarray], default=None
            Normalized targets used in dual mode.
        """
        # Update primary metrics
        for metric in self.metrics.values():
            metric.update(predictions, targets)

        # Update normalized metrics if dual mode is active
        if self.is_dual and norm_preds is not None and norm_targets is not None:
            for metric in self.normalized_metrics.values():
                metric.update(norm_preds, norm_targets)

    def compute(self) -> Dict[str, float]:
        """
        Compute and return all managed metric values.

        Returns
        -------
        Dict[str, float]
            Dictionary of computed metric values.
        """
        results = {}

        # Classification results (usually no suffixes needed)
        if self.task_type == "classification":
            return {name: metric.compute() for name, metric in self.metrics.items()}

        # Regression / RUL / forecasting: dual-suffix semantics match PHM evaluators
        suffix = "denormalized" if self.is_dual else "normalized"
        for name, metric in self.metrics.items():
            results[f"{name}_{suffix}"] = metric.compute()

        if self.is_dual and self.normalized_metrics:
            for name, metric in self.normalized_metrics.items():
                results[f"{name}_normalized"] = metric.compute()

        return results

    def reset(self) -> None:
        """Reset all internal metric states."""
        for metric in self.metrics.values():
            metric.reset()
        if self.normalized_metrics:
            for metric in self.normalized_metrics.values():
                metric.reset()
