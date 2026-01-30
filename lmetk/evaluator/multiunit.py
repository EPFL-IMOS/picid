import logging
from typing import Any, Dict, List, Optional, override
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr

from lmetk.evaluator.default import DefaultEvaluator
from lmetk.baselines.definitions import REGRESSION_TASKS

logger = logging.getLogger(__name__)


class MultiUnitEvaluator(DefaultEvaluator):
    """
    Evaluator that splits metrics by unit ID and provides per-unit trend plotting.
    Strictly for Regression / RUL tasks.
    """

    def __init__(
        self,
        metric_names: List[str],
        log_image: bool = True,
        log_image_every_n_epochs: int = 10,
        enable_plot_subsampling: bool = True,
        plot_subsample_threshold: int = 2000,
        plot_subsample_factor: int = 10,
        log_per_unit_metrics: bool = True,
        remote_logger: Optional[Any] = None,
        **kwargs,
    ):
        # Enforce Regression Task Type
        task_type = kwargs.get("task_type", "regression")
        assert (
            task_type in REGRESSION_TASKS
        ), f"MultiUnitEvaluator only supports regression tasks, got '{task_type}'"

        super().__init__(
            metric_names=metric_names, remote_logger=remote_logger, **kwargs
        )

        self.log_fit_vs_target_curves = log_image
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.enable_plot_subsampling = enable_plot_subsampling
        self.plot_subsample_threshold = plot_subsample_threshold
        self.plot_subsample_factor = plot_subsample_factor
        self.log_per_unit_metrics = log_per_unit_metrics

        self.unit_metrics = defaultdict(self.metric_manager.create_metric_set)
        self.normalized_unit_metrics = (
            defaultdict(self.metric_manager.create_metric_set)
            if self.scaling_wrapper.apply_inverse
            else None
        )

    @override
    def reset(self):
        super().reset()
        self.unit_metrics.clear()
        if self.normalized_unit_metrics is not None:
            self.normalized_unit_metrics.clear()

    @override
    def update(self, model_out: dict):
        if not self._validate_input(model_out):
            return
        super().update(model_out)
        batch = self._prepare_batch_data(model_out)
        u_ids = model_out["unit_id"]
        unique_units, get_indices, get_key = self._get_unit_iteration_tools(u_ids)

        for u in unique_units:
            k, idx = get_key(u), get_indices(u)
            for m in self.unit_metrics[k].values():
                m.update(batch["preds"][idx], batch["targets"][idx])

            if self.normalized_unit_metrics is not None:
                for m in self.normalized_unit_metrics[k].values():
                    m.update(batch["norm_preds"][idx], batch["norm_targets"][idx])

    # === FIX: OVERRIDE COMPUTE TO INJECT PLOTTING ===
    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Orchestrates computation and optional plotting.
        """
        # Logic: Plot only if configured and frequency condition is met
        if self.log_fit_vs_target_curves:
            is_valid_mode = mode != "train"
            is_valid_epoch = epoch % self.log_image_every_n_epochs == 0

            # Plot for validation/test modes, adhering to epoch frequency for val
            if is_valid_mode and (mode != "val" or is_valid_epoch):
                self._log_pred_vs_target_curves(mode, step)

        return super().compute(mode, epoch, step)

    # ================================================

    def _get_unit_iteration_tools(self, u_ids):
        if u_ids.ndim == 1:
            return np.unique(u_ids), lambda u: np.where(u_ids == u)[0], lambda u: int(u)
        if u_ids.ndim == 2:
            return (
                np.unique(u_ids, axis=0),
                lambda u: np.where((u_ids == u).all(axis=1))[0],
                lambda u: tuple(int(x) for x in u),
            )
        raise ValueError("Only 1D and 2D are supported")

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        res, collector = {}, defaultdict(list)
        is_dual = self.scaling_wrapper.apply_inverse
        suffix = "denormalized" if is_dual else "normalized"

        # 1. Collect Unit Metrics
        for u_id, metrics in self.unit_metrics.items():
            for n, m in metrics.items():
                val = m.compute()
                if self.log_per_unit_metrics:
                    res[f"{n}_{u_id}_{suffix}"] = val
                collector[f"{n}_{suffix}"].append(val)

        if self.normalized_unit_metrics is not None:
            for u_id, metrics in self.normalized_unit_metrics.items():
                for n, m in metrics.items():
                    val = m.compute()
                    if self.log_per_unit_metrics:
                        res[f"{n}_{u_id}_normalized"] = val
                    collector[f"{n}_normalized"].append(val)

        # 2. Compute Aggregate Means
        for n in self.metric_names:

            def _set_mean(key_suffix):
                key = f"{n}_{key_suffix}"
                if key in collector and collector[key]:
                    res[f"{key}_mean"] = np.mean(collector[key])
                else:
                    res[f"{key}_mean"] = float("nan")

            _set_mean(suffix)
            if self.normalized_unit_metrics is not None:
                _set_mean("normalized")

        res.update(super()._compute_metrics())
        return res

    @override
    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        res = super()._prepare_predictions_for_save()
        if self.collect_predictions and self.buffer.unit_ids:
            u_ids = np.concatenate(self.buffer.unit_ids, axis=0)
            u_dims = ["sample"] + [f"unit_dim{i}" for i in range(1, u_ids.ndim)]
            res["unit_ids"] = xr.DataArray(data=u_ids, dims=u_dims)
        return res

    def _log_pred_vs_target_curves(self, mode: str, step: int) -> None:
        if (
            not self.remote_logger
            or not self.collect_predictions
            or not self.buffer.preds
        ):
            return
        is_dual = self.scaling_wrapper.apply_inverse
        p_all = np.concatenate(
            self.buffer.scaled_preds if is_dual else self.buffer.preds, axis=0
        )
        t_all = np.concatenate(
            self.buffer.scaled_targets if is_dual else self.buffer.targets, axis=0
        )
        u_all = np.concatenate(self.buffer.unit_ids, axis=0)

        unique_units, get_indices, get_key = self._get_unit_iteration_tools(u_all)
        for u in unique_units:
            idx, unit_key = get_indices(u), get_key(u)
            p, t = p_all[idx].flatten(), t_all[idx].flatten()
            if self.enable_plot_subsampling and p.size > self.plot_subsample_threshold:
                p, t = (
                    p[:: self.plot_subsample_factor],
                    t[:: self.plot_subsample_factor],
                )

            fig, ax = plt.subplots()
            ax.plot(t, "--", label="Target")
            ax.plot(p, label="Prediction")
            ax.set_title(f"Unit {unit_key}")
            self.log_plot(fig, f"pred_vs_target_unit_{unit_key}", mode, 0, step)
            plt.close(fig)
