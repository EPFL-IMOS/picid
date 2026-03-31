from typing import Any, Dict, List
import numpy as np
from picid.evaluator.default import DefaultEvaluator
from picid.metrics.manager import MetricManager


class MultiUnitEvaluator(DefaultEvaluator):
    """
    Evaluator that calculates metrics individually per unit ID.

    Parameters
    ----------
    metric_names : List[str]
        Metrics to track.
    log_per_unit_metrics : bool, default=True
        Whether to log metrics per unit ID.
    **kwargs : Any
        Additional evaluator keyword arguments.
    """

    def __init__(
        self,
        metric_names: List[str],
        log_per_unit_metrics: bool = True,
        **kwargs: Any,
    ):
        super().__init__(metric_names=metric_names, **kwargs)
        self.log_per_unit_metrics = log_per_unit_metrics
        self.unit_managers: Dict[Any, MetricManager] = {}

    def update(self, model_out: Dict[str, Any]) -> None:
        """
        Update global metrics and per-unit trackers.

        Parameters
        ----------
        model_out : Dict[str, Any]
            Dictionary containing predictions, targets, and unit IDs.
        """
        if not self._validate_input(model_out):
            return
        super().update(model_out)

        batch = self._prepare_batch_data(model_out)
        u_ids = model_out["unit_id"]
        unique_units, get_indices, get_key = self._get_unit_iteration_tools(u_ids)

        for u in unique_units:
            k, idx = get_key(u), get_indices(u)
            if k not in self.unit_managers:
                self.unit_managers[k] = self._create_configured_manager()

            self.unit_managers[k].update(
                predictions=batch["preds"][idx],
                targets=batch["targets"][idx],
                norm_preds=(
                    batch["norm_preds"][idx]
                    if batch["norm_preds"] is not None
                    else None
                ),
                norm_targets=(
                    batch["norm_targets"][idx]
                    if batch["norm_targets"] is not None
                    else None
                ),
            )

    def _create_configured_manager(self) -> MetricManager:
        """
        Create a manager that mirrors the primary manager configuration.

        Returns
        -------
        MetricManager
            Configured metric manager for a single unit.
        """
        return MetricManager(
            metric_names=self.metric_manager.metric_names,
            task_type=self.task_type,
            paths=self.paths,
            num_classes=self.num_classes,
            is_dual=self.metric_manager.is_dual,
        )

    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Aggregate unit-level metrics into a global result set.

        Parameters
        ----------
        mode : str
            Evaluation mode.
        epoch : int
            Current epoch index.
        step : int
            Current step index.

        Returns
        -------
        Dict[str, float]
            Aggregated metric values.
        """
        res = super().compute(mode, epoch, step)
        from collections import defaultdict

        collector = defaultdict(list)
        for u_id, manager in self.unit_managers.items():
            unit_results = manager.compute()
            for key, val in unit_results.items():
                if self.log_per_unit_metrics:
                    res[f"{key}_{u_id}"] = val
                collector[key].append(val)

        for key, vals in collector.items():
            res[f"{key}_mean"] = np.mean(vals) if vals else float("nan")
        return res

    def _get_unit_iteration_tools(self, u_ids: np.ndarray) -> Any:
        """
        Build helpers for iterating over 1D or 2D unit identifiers.

        Parameters
        ----------
        u_ids : np.ndarray
            Array of unit identifiers.

        Returns
        -------
        Any
            Unique identifiers, index lookup function, and key conversion function.
        """
        if u_ids.ndim == 1:
            return np.unique(u_ids), lambda u: np.where(u_ids == u)[0], lambda u: int(u)
        return (
            np.unique(u_ids, axis=0),
            lambda u: np.where((u_ids == u).all(axis=1))[0],
            lambda u: tuple(int(x) for x in u),
        )

    def reset(self) -> None:
        """Reset internal state for a new evaluation run."""
        super().reset()
        self.unit_managers.clear()  # ADDED: Crucial for test passing and memory safety
