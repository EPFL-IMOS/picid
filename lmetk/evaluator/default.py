import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, override
from collections import defaultdict
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from lmetk.evaluator.base import AbstractEvaluator
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.baselines.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    FORECASTING_TASKS,
    ALL_TASKS,
)

logger = logging.getLogger(__name__)


class MetricManager:
    """Handles logic for primary and normalized metric sets."""

    def __init__(
        self,
        metric_names: List[str],
        task_type: str,
        paths: Any,
        num_classes: int,
        is_dual: bool,
    ):
        self.metric_names = metric_names
        self.task_type = task_type
        self.paths = paths
        self.num_classes = num_classes
        self.is_dual = is_dual

        # Initialize global metrics
        self.metrics = self.create_metric_set()
        self.normalized_metrics = self.create_metric_set() if is_dual else None

    def create_metric_set(self) -> Dict:
        """Helper to create a fresh set of metrics using the Factory."""
        res = {}
        for n in self.metric_names:
            n_low = n.lower()
            if self.task_type in CLASSIFICATION_TASKS:
                res[n_low] = MetricFactory.create_classification_metric(
                    n_low, self.num_classes
                )
            else:
                res[n_low] = MetricFactory.create_metric(n_low, self.paths)
        return res

    def update(
        self,
        p: np.ndarray,
        t: np.ndarray,
        p_norm: Optional[np.ndarray] = None,
        t_norm: Optional[np.ndarray] = None,
    ):
        for m in self.metrics.values():
            m.update(p, t)
        if self.normalized_metrics and p_norm is not None:
            for m in self.normalized_metrics.values():
                m.update(p_norm, t_norm)

    def compute(self) -> Dict[str, float]:
        res = {}
        suffix = "denormalized" if self.is_dual else "normalized"
        for n, m in self.metrics.items():
            key = n if self.task_type in CLASSIFICATION_TASKS else f"{n}_{suffix}"
            res[key] = m.compute()
        if self.is_dual:
            for n, m in self.normalized_metrics.items():
                res[f"{n}_normalized"] = m.compute()
        return res

    def reset(self):
        for m in self.metrics.values():
            m.reset()
        if self.normalized_metrics:
            for m in self.normalized_metrics.values():
                m.reset()


class PredictionBuffer:
    """Handles raw data accumulation and formatting."""

    def __init__(self):
        self.preds, self.targets = [], []
        self.scaled_preds, self.scaled_targets = [], []
        self.unit_ids = []

    def accumulate(self, batch: Dict[str, Any], unit_id: Optional[np.ndarray] = None):
        is_dual = batch.get("is_dual", False)
        self.preds.append(batch["norm_preds"] if is_dual else batch["preds"])
        self.targets.append(batch["norm_targets"] if is_dual else batch["targets"])
        if is_dual:
            self.scaled_preds.append(batch["preds"])
            self.scaled_targets.append(batch["targets"])
        if unit_id is not None:
            self.unit_ids.append(unit_id)

    def prepare_dataset(self, task_type: str, is_dual: bool) -> Dict[str, Any]:
        if not self.preds:
            return {}
        p_src = self.scaled_preds if is_dual else self.preds
        t_src = self.scaled_targets if is_dual else self.targets
        p, t = np.concatenate(p_src, axis=0), np.concatenate(t_src, axis=0)

        if task_type in CLASSIFICATION_TASKS and p.shape[-1] > 1:
            p = np.argmax(p, axis=-1, keepdims=True)

        d_names = ["sample"] + [f"dim{i}" for i in range(1, p.ndim)]
        res = {"preds": (d_names, p), "targets": (d_names, t)}
        if is_dual:
            res["preds_normalized"] = (d_names, np.concatenate(self.preds, axis=0))
            res["targets_normalized"] = (d_names, np.concatenate(self.targets, axis=0))
        if self.unit_ids:
            u = np.concatenate(self.unit_ids, axis=0)
            u_dims = ["sample"] + [f"unit_dim{i}" for i in range(1, u.ndim)]
            res["unit_ids"] = (u_dims, u)
        return res

    def clear(self):
        for coll in [
            self.preds,
            self.targets,
            self.scaled_preds,
            self.scaled_targets,
            self.unit_ids,
        ]:
            coll.clear()


class DefaultEvaluator(AbstractEvaluator):
    def __init__(
        self, metric_names: List[str], task_type: str = "regression", **kwargs
    ):

        # 1. Validation: Metrics
        if not metric_names:
            raise ValueError("metric_names list cannot be empty.")

        super().__init__(**kwargs)
        self.task_type = task_type.lower()

        assert self.task_type in ALL_TASKS, f"Unsupported task_type: {self.task_type}"

        self.metric_names = [n.lower() for n in metric_names]
        self.num_classes = kwargs.get("num_classes")
        self.save_predictions = kwargs.get("save_predictions", False)
        self.collect_predictions = self.save_predictions or kwargs.get(
            "collect_predictions", True
        )
        self.remote_logger = kwargs.get("remote_logger")

        self.buffer = PredictionBuffer()
        self.metric_manager = MetricManager(
            self.metric_names,
            self.task_type,
            self.paths,
            self.num_classes,
            self.scaling_wrapper.apply_inverse,
        )

    @override
    def update(self, model_out: dict):
        if not self._validate_input(model_out):
            return
        batch = self._prepare_batch_data(model_out)
        if self.collect_predictions:
            self.buffer.accumulate(batch, model_out.get("unit_id"))
        self.metric_manager.update(
            batch["preds"],
            batch["targets"],
            batch.get("norm_preds"),
            batch.get("norm_targets"),
        )

    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        if self.save_predictions:
            self._save_predictions_to_file(self._prepare_predictions_for_save(), mode)
        return self._compute_metrics()

    @override
    def reset(self):
        self.buffer.clear()
        self.metric_manager.reset()

    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        return self.buffer.prepare_dataset(
            self.task_type, self.scaling_wrapper.apply_inverse
        )

    def _compute_metrics(self) -> Dict[str, float]:
        return self.metric_manager.compute()

    def _validate_input(self, model_out: dict) -> bool:
        p, t = model_out.get("predictions"), model_out.get("targets")
        return not (p is None or t is None or p.shape[0] == 0)

    def _prepare_batch_data(self, model_out: dict) -> Dict[str, Any]:
        preds, targets = model_out["predictions"], model_out["targets"]
        is_dual = self.scaling_wrapper.apply_inverse
        res = {
            "is_dual": is_dual,
            "norm_preds": preds.copy() if is_dual else None,
            "norm_targets": targets.copy() if is_dual else None,
        }
        if self.task_type in CLASSIFICATION_TASKS:
            p, t = preds, targets.astype(int)
        else:
            p, t = self.scaling_wrapper.inverse_transform_if_needed(
                preds, targets, metadata={"unit_id": model_out.get("unit_id")}
            )
        res.update({"preds": p, "targets": t})
        return res

    def _save_predictions_to_file(self, data: Dict[str, Any], mode: str):
        if not data or not self.paths:
            return
        p_dir = getattr(self.paths, "eval_details", None)
        if p_dir:
            out_path = Path(p_dir) / mode
            out_path.mkdir(parents=True, exist_ok=True)
            xr.Dataset(data).to_netcdf(out_path / "predictions.nc")

    def log_plot(self, fig: plt.Figure, title: str, mode: str, epoch: int, step: int):
        if self.remote_logger:
            self.remote_logger.experiment.log({f"plots/{mode}/{title}": fig})
