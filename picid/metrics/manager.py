import logging
from typing import Any, Dict, List, Optional

import numpy as np
from picid.metrics.metric_factory import MetricFactory

logger = logging.getLogger(__name__)


class MetricManager:
    """
    Orchestrates multiple metric instances.
    Handles dual-scaling (normalized/denormalized) and task-specific aggregation.
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
        Args:
            metric_names: List of metrics to track (e.g., ['mse', 'mae']).
            task_type: 'regression' or 'classification'.
            paths: Required for AbstractFileSystemMetrics (e.g., Railway metrics).
            num_classes: Required for classification metrics.
            is_dual: If True, tracks both raw and normalized metrics.
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
        """Uses MetricFactory to build the required metric objects."""
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
                # Passes paths for regression/railway metrics
                res[name] = MetricFactory.create_metric(name, self.paths or {})
        return res

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        norm_preds: Optional[np.ndarray] = None,
        norm_targets: Optional[np.ndarray] = None,
    ) -> None:
        """Updates all managed metrics with new batch data."""
        # Update primary metrics
        for metric in self.metrics.values():
            metric.update(predictions, targets)

        # Update normalized metrics if dual mode is active
        if self.is_dual and norm_preds is not None and norm_targets is not None:
            for metric in self.normalized_metrics.values():
                metric.update(norm_preds, norm_targets)

    def compute(self) -> Dict[str, float]:
        """Computes and returns a dictionary of all metric results."""
        results = {}

        # Classification results (usually no suffixes needed)
        if self.task_type == "classification":
            return {name: metric.compute() for name, metric in self.metrics.items()}

        # Regression / PHM results with dual-scaling suffixes
        suffix = "denormalized" if self.is_dual else "normalized"
        for name, metric in self.metrics.items():
            results[f"{name}_{suffix}"] = metric.compute()

        if self.is_dual and self.normalized_metrics:
            for name, metric in self.normalized_metrics.items():
                results[f"{name}_normalized"] = metric.compute()

        return results

    def reset(self) -> None:
        """Resets all internal metric states."""
        for metric in self.metrics.values():
            metric.reset()
        if self.normalized_metrics:
            for metric in self.normalized_metrics.values():
                metric.reset()
