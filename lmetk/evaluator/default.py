import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, override
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from lmetk.evaluator.base import AbstractEvaluator
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.baselines.definitions import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    FORECASTING_TASKS,
    ALL_TASKS,
)

logger = logging.getLogger(__name__)


class DefaultEvaluator(AbstractEvaluator):
    """
    Unified evaluator for regression and classification.
    Strictly enforces PHM shapes (N, T, 1) and breaks on empty computation.
    """

    def __init__(
        self,
        metric_names: List[str],
        task_type: str = "regression",
        num_classes: Optional[int] = None,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[Any] = None,
        remote_logger: Optional[Any] = None,
        collect_predictions: bool = True,
        **kwargs,
    ):
        if not metric_names:
            raise ValueError("metric_names cannot be empty")

        super().__init__(
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            paths=paths,
            **kwargs,
        )

        self.inverse_transform = inverse_transform
        self.task_type = task_type.lower()
        assert self.task_type in ALL_TASKS, f"Unsupported task_type: {self.task_type}"

        self.metric_names = [name.lower() for name in metric_names]
        self.num_classes = num_classes
        self.save_predictions = save_predictions
        self.collect_predictions = save_predictions or collect_predictions
        self.remote_logger = remote_logger
        self.out_path = None

        self.metrics = self._init_metric_dict()
        if self.scaling_wrapper.apply_inverse:
            self.normalized_metrics = self._init_metric_dict()

        if self.collect_predictions:
            self.preds, self.targets = [], []
            self.scaled_preds, self.scaled_targets = [], []

    @override
    def update(self, model_out: dict):
        if not self._validate_input(model_out):
            return

        batch = self._prepare_batch_data(model_out)

        if self.collect_predictions:
            self._accumulate_batch(batch)

        self._update_metric_set(self.metrics, batch["preds"], batch["targets"])

        if batch["is_dual"]:
            self._update_metric_set(
                self.normalized_metrics, batch["norm_preds"], batch["norm_targets"]
            )

    def _validate_input(self, model_out: dict) -> bool:
        preds, targets = model_out.get("predictions"), model_out.get("targets")
        if preds is None or targets is None or preds.shape[0] == 0:
            return False
        return True

    def _prepare_batch_data(self, model_out: dict) -> Dict[str, Any]:
        preds, targets = model_out["predictions"], model_out["targets"]
        is_dual = self.scaling_wrapper.apply_inverse

        res = {
            "is_dual": is_dual,
            "norm_preds": preds.copy() if is_dual else None,
            "norm_targets": targets.copy() if is_dual else None,
        }

        if self.task_type in CLASSIFICATION_TASKS:
            assert (
                preds.ndim == 3 and preds.shape[2] == self.num_classes
            ), f"Expected prediction shape (N, T, {self.num_classes}), got {preds.shape}"
            p, t = preds, targets.astype(int)

        elif self.task_type in REGRESSION_TASKS or self.task_type in FORECASTING_TASKS:
            assert (
                targets.ndim == 3 and targets.shape[2] == 1
            ), f"Expected target shape (N, T, 1), got {targets.shape}"
            assert (
                preds.ndim == 3 and preds.shape[2] == 1
            ), f"Expected prediction shape (N, T, 1), got {preds.shape}"

            p, t = self.scaling_wrapper.inverse_transform_if_needed(
                preds, targets, metadata={"unit_id": model_out.get("unit_id")}
            )
        else:
            raise ValueError(f"Evaluation does not support task_type: {self.task_type}")

        res.update({"preds": p, "targets": t})
        return res

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        res = {}
        is_dual = self.scaling_wrapper.apply_inverse
        p_suffix = self._get_metric_suffix(is_primary=True)

        for n, m in self.metrics.items():
            key = f"{n}_{p_suffix}" if p_suffix else n
            res[key] = m.compute()

        if is_dual:
            for n, m in self.normalized_metrics.items():
                res[f"{n}_normalized"] = m.compute()
        return res

    def _get_metric_suffix(self, is_primary: bool) -> Optional[str]:
        if self.task_type in CLASSIFICATION_TASKS:
            return None
        return (
            "denormalized"
            if (is_primary and self.scaling_wrapper.apply_inverse)
            else "normalized"
        )

    def _init_metric_dict(self) -> Dict:
        return {
            n: (
                MetricFactory.create_classification_metric(n, self.num_classes)
                if self.task_type in CLASSIFICATION_TASKS
                else MetricFactory.create_metric(n, paths=self.paths)
            )
            for n in self.metric_names
        }

    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        if self.save_predictions:
            self._save_predictions_to_file(self._prepare_predictions_for_save(), mode)
        return self._compute_metrics()

    @override
    def reset(self):
        if hasattr(self, "preds") and self.preds is not None:
            self.preds.clear()
            self.targets.clear()
            self.scaled_preds.clear()
            self.scaled_targets.clear()
        for m in self.metrics.values():
            m.reset()
        if hasattr(self, "normalized_metrics"):
            for m in self.normalized_metrics.values():
                m.reset()

    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        if not hasattr(self, "preds") or not self.preds:
            return {}
        is_dual = self.scaling_wrapper.apply_inverse
        p = np.concatenate(self.scaled_preds if is_dual else self.preds, axis=0)
        t = np.concatenate(self.scaled_targets if is_dual else self.targets, axis=0)

        if self.task_type in CLASSIFICATION_TASKS and p.shape[-1] > 1:
            p = np.argmax(p, axis=-1, keepdims=True)

        d_names = ["sample"] + [f"dim{i}" for i in range(1, p.ndim)]
        res = {"preds": (d_names, p), "targets": (d_names, t)}
        if is_dual:
            res["preds_normalized"] = (d_names, np.concatenate(self.preds, axis=0))
            res["targets_normalized"] = (d_names, np.concatenate(self.targets, axis=0))
        return res

    def _save_predictions_to_file(self, data: Dict[str, Any], mode: str):
        if not data or not self.paths:
            return
        p_dir = getattr(self.paths, "eval_details", None)
        if p_dir:
            self.out_path = Path(p_dir) / mode
            self.out_path.mkdir(parents=True, exist_ok=True)
            xr.Dataset(data).to_netcdf(self.out_path / "predictions.nc")

    def _accumulate_batch(self, batch: Dict[str, Any]):
        sp = batch["norm_preds"] if batch["is_dual"] else batch["preds"]
        st = batch["norm_targets"] if batch["is_dual"] else batch["targets"]
        self.preds.append(sp)
        self.targets.append(st)
        if batch["is_dual"]:
            self.scaled_preds.append(batch["preds"])
            self.scaled_targets.append(batch["targets"])

    def _update_metric_set(self, m_dict: Dict, p: np.ndarray, t: np.ndarray):
        for m in m_dict.values():
            m.update(p, t)

    def log_plot(self, fig: plt.Figure, title: str, mode: str, epoch: int, step: int):
        if self.remote_logger:
            self.remote_logger.experiment.log({f"plots/{mode}/{title}": fig})
