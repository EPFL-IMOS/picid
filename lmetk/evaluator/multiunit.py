import logging
from typing import Any, Dict, List, Optional, override
from collections import defaultdict
import numpy as np
import xarray as xr

# Import precisely to match legacy @patch calls in test suite
import matplotlib
import matplotlib.pyplot as plt

from lmetk.baselines.definitions import REGRESSION_TASKS
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.evaluator.default import DefaultEvaluator

logger = logging.getLogger(__name__)


class MultiUnitEvaluator(DefaultEvaluator):
    """
    Regression evaluator managing per-unit metrics.
    Uses unified DefaultEvaluator helpers to maintain scaling and collection logic.
    """

    def __init__(
        self,
        metric_names: List[str],
        inverse_transform: Optional[Any] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[Any] = None,
        remote_logger: Optional[Any] = None,
        collect_predictions: bool = True,
        log_image_every_n_epochs: int = 10,
        log_image: bool = True,
        enable_plot_subsampling: bool = True,
        plot_subsample_threshold: int = 2000,
        plot_subsample_factor: int = 10,
        log_per_unit_metrics: bool = True,
        **kwargs,
    ):

        # Pre-init specific collections to satisfy child reset() logic during super().__init__
        self.metric_names = [n.lower() for n in metric_names]
        self.paths = paths
        self.unit_metrics = defaultdict(self._create_metric_dict)
        if apply_inverse_scaling:
            self.normalized_unit_metrics = defaultdict(self._create_metric_dict)

        super().__init__(
            metric_names=metric_names,
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            paths=paths,
            remote_logger=remote_logger,
            collect_predictions=collect_predictions,
            **kwargs,
        )

        assert self.task_type in REGRESSION_TASKS, "Unsupported task_type"

        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.log_fit_vs_target_curves = log_image
        self.enable_plot_subsampling = enable_plot_subsampling
        self.plot_subsample_threshold = plot_subsample_threshold
        self.plot_subsample_factor = plot_subsample_factor
        self.log_per_unit_metrics = log_per_unit_metrics

        if self.collect_predictions:
            self.unit_ids = []

    def _create_metric_dict(self) -> Dict:
        return {
            n: MetricFactory.create_metric(n, self.paths) for n in self.metric_names
        }

    @override
    def reset(self):
        super().reset()
        if hasattr(self, "unit_ids"):
            self.unit_ids.clear()
        if hasattr(self, "unit_metrics"):
            self.unit_metrics.clear()
        if hasattr(self, "normalized_unit_metrics"):
            self.normalized_unit_metrics.clear()

    @override
    def update(self, model_out: dict):
        if not self._validate_input(model_out):
            return

        batch = self._prepare_batch_data(model_out)
        u_ids = model_out["unit_id"]

        if self.collect_predictions:
            self._accumulate_batch(batch)
            self.unit_ids.append(u_ids)

        unique_units, get_indices, get_key = self._get_unit_iteration_tools(u_ids)
        for u in unique_units:
            k, idx = get_key(u), get_indices(u)
            self._update_metric_set(
                self.unit_metrics[k], batch["preds"][idx], batch["targets"][idx]
            )
            if batch["is_dual"]:
                self._update_metric_set(
                    self.normalized_unit_metrics[k],
                    batch["norm_preds"][idx],
                    batch["norm_targets"][idx],
                )

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
    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        res = super()._prepare_predictions_for_save()
        if not res or not hasattr(self, "unit_ids") or not self.unit_ids:
            return res
        u_ids = np.concatenate(self.unit_ids, axis=0)
        u_dims = ["sample"] + [f"unit_dim{i}" for i in range(1, u_ids.ndim)]
        res["unit_ids"] = xr.DataArray(data=u_ids, dims=u_dims)
        return res

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        if not self.unit_metrics:
            return {f"{n}_mean": 0.0 for n in self.metric_names}
        res, collector = {}, defaultdict(list)
        is_dual = self.scaling_wrapper.apply_inverse
        p_suffix = "denormalized" if is_dual else "normalized"

        for u_id, metrics in self.unit_metrics.items():
            for n, m in metrics.items():
                val = self._safe_compute(m)
                if self.log_per_unit_metrics:
                    res[f"{n}_{u_id}_{p_suffix}"] = val
                collector[f"{n}_{p_suffix}"].append(val)

        if is_dual:
            for u_id, metrics in self.normalized_unit_metrics.items():
                for n, m in metrics.items():
                    val = self._safe_compute(m)
                    if self.log_per_unit_metrics:
                        res[f"{n}_{u_id}_normalized"] = val
                    collector[f"{n}_normalized"].append(val)

        for k, vals in collector.items():
            res[f"{k}_mean"] = np.mean(vals)
        return res

    @override
    def compute(
        self, mode: str = "val", epoch: int = 0, step: int = 0
    ) -> Dict[str, float]:
        if self.log_fit_vs_target_curves:
            if mode != "train" or (
                mode == "val" and epoch % self.log_image_every_n_epochs == 0
            ):
                self._log_pred_vs_target_curves(mode, step)
        return super().compute(mode, epoch, step)

    # --- Test Logic & Visualization Compatibility ---

    def _inverse_transform_per_unit(self, predictions, targets, unit_ids):
        if unit_ids.ndim not in (1, 2):
            logger.warning("Unsupported shape")
            return self.scaling_wrapper.inverse_transform_if_needed(
                predictions, targets
            )
        res_p, res_t = self.scaling_wrapper.inverse_transform_if_needed(
            predictions, targets, metadata={"unit_id": unit_ids}
        )
        if self.scaling_wrapper.apply_inverse and np.isinf(res_p).any():
            raise ValueError("not filled after inverse transform")
        return res_p, res_t

    def _log_pred_vs_target_curves(self, mode: str, step: int) -> None:
        """
        Log prediction vs target curves for each unit using remote logger.

        PHM Context: Visual verification of RUL degradation curves is essential
        for diagnosing model performance across different equipment units.
        """
        # 1. Safety Guard for empty data or missing logger
        if not self.remote_logger or not hasattr(self, "preds") or not self.preds:
            return

        try:
            # 2. Concatenate all collected batches
            p_all = np.concatenate(self.preds, axis=0)
            t_all = np.concatenate(self.targets, axis=0)
            u_all = np.concatenate(self.unit_ids, axis=0)

            # 3. Get iteration tools (handles 1D and 2D unit IDs)
            unique_units, get_indices, get_key = self._get_unit_iteration_tools(u_all)
        except (ValueError, Exception) as e:
            # Matches Requirement MU-LOG-2 (logs warning on unsupported shapes)
            logger.warning(f"Unsupported shape: {str(e)}")
            return

        # 4. Iterate and Plot each unique Unit
        for u in unique_units:
            idx = get_indices(u)
            unit_key = get_key(u)  # Formats 2D as tuple (2, 3), 1D as int 2

            # Flatten for time-series plotting
            p = p_all[idx].flatten()
            t = t_all[idx].flatten()

            # 5. Apply Subsampling (Requirement MU-PLOT-2)
            title_suffix = ""
            if self.enable_plot_subsampling and p.size > self.plot_subsample_threshold:
                p = p[:: self.plot_subsample_factor]
                t = t[:: self.plot_subsample_factor]
                title_suffix = f" (Subsampled 1:{self.plot_subsample_factor})"

            # 6. Generate Figure using clean 'plt' alias for mock tracking
            fig, ax = plt.subplots()
            ax.plot(t, "--", label="Target", alpha=0.7)
            ax.plot(p, label="Prediction", alpha=0.9)

            # FIX: Use unit_key for the title to ensure tuple format (e.g. "Unit (2, 3)")
            ax.set_title(f"Unit {unit_key}{title_suffix}")

            ax.set_xlabel("Time steps")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 7. Log and Cleanup
            self.log_plot(
                fig=fig,
                title=f"pred_vs_target_unit_{unit_key}",
                mode=mode,
                epoch=0,
                step=step,
            )
            plt.close(fig)
