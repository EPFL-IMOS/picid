import logging
from typing import Any, Dict, List, Optional, override
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib

# Assuming these imports are correct for your project structure:
from lmetk.baselines.definitions import REGRESSION_TASKS
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.metrics.base import AbstractMetric
from lmetk.evaluator.default import DefaultEvaluator  # Assumed base class


logger = logging.getLogger(__name__)


class MultiUnitEvaluator(DefaultEvaluator):
    """
    Evaluator that handles per-unit metrics, inverse scaling, and prediction saving.
    It inherits core orchestration logic from DefaultEvaluator and uses dual metric
    storage to report both normalized and denormalized results without modifying
    the base metric classes.
    """

    def __init__(
        self,
        metric_names: List[str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[str] = None,
        remote_logger: Optional[Any] = None,
        collect_predictions: bool = False,
        log_image_every_n_epochs: bool = 10,
        log_image: bool = True,
        enable_plot_subsampling: bool = True,
        plot_subsample_threshold: int = 2000,
        plot_subsample_factor: int = 10,
        log_per_unit_metrics: bool = True,
        **kwargs,
    ):
        # 1. Initialize Base
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

        assert (
            self.task_type in REGRESSION_TASKS
        ), f"Unsupported task_type: {self.task_type}"

        # 2. MultiUnit Specific Params
        self.log_image_every_n_epochs = log_image_every_n_epochs
        self.log_fit_vs_target_curves = log_image
        self.enable_plot_subsampling = enable_plot_subsampling
        self.plot_subsample_threshold = plot_subsample_threshold
        self.plot_subsample_factor = plot_subsample_factor

        # 🎯 Store the new parameter
        self.log_per_unit_metrics = log_per_unit_metrics

        # 3. Extra Collections (Base only handles preds/targets/scaled_preds/scaled_targets)
        if self.collect_predictions:
            self.unit_ids = []

        # 4. Per-Unit Metric Storage
        # Primary storage (stores Denormalized if applied_inverse_scaling=True)
        self.unit_metrics = defaultdict(self._create_metric_dict)

        # Secondary storage for the other set of results (stores Normalized if applied_inverse_scaling=True)
        if self.scaling_wrapper.apply_inverse:
            self.normalized_unit_metrics = defaultdict(self._create_metric_dict)

        self.reset()

    def _create_metric_dict(self) -> Dict[str, AbstractMetric]:
        """Factory function to create a dictionary of new metric instances."""
        return {
            name: MetricFactory.create_metric(name, self.paths)
            for name in self.metric_names
        }

    @override
    def reset(self):
        """Reset base collections and multi-unit specific collections."""
        super().reset()
        if self.collect_predictions:
            self.unit_ids = []
        self.unit_metrics.clear()
        if hasattr(self, "normalized_unit_metrics"):
            self.normalized_unit_metrics.clear()

    @override
    def update(self, model_out: dict):
        """
        Update metrics with new predictions, handling unit splitting, scaling, and DUAL metric update.
        """
        predictions = model_out["predictions"]
        targets = model_out["targets"]
        unit_ids = model_out["unit_id"]

        is_denormalized_run = self.scaling_wrapper.apply_inverse

        if self.task_type in REGRESSION_TASKS:
            assert (
                predictions.ndim == 3 and predictions.shape[2] == 1
            ), f"Expected shape (N, T, 1), got {targets.shape}"
            predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
                predictions, targets
            )

        # Data BEFORE inverse scaling (normalized data) - Must copy before modification
        if is_denormalized_run:
            normalized_predictions = predictions.copy()
            normalized_targets = targets.copy()

        # Collect original data and unit IDs (Base logic uses predictions/targets here)
        if self.collect_predictions:
            self.preds.append(predictions)
            self.targets.append(targets)
            self.unit_ids.append(unit_ids)

        # Apply inverse scaling (MODIFIES predictions/targets in place)
        predictions, targets = self._inverse_transform_per_unit(
            predictions, targets, unit_ids
        )

        # Collect scaled data (if inverse scaling was applied)
        if self.collect_predictions and is_denormalized_run:
            self.scaled_preds.append(predictions)
            self.scaled_targets.append(targets)

        # --- Per-Unit Metric Update Logic ---

        if unit_ids.ndim == 1:
            unique_units = np.unique(unit_ids)
            get_indices = lambda unit: np.where(unit_ids == unit)[0]
            get_key = lambda unit: int(unit[0])
        elif unit_ids.ndim == 2:
            unique_units = np.unique(unit_ids, axis=0)
            get_indices = lambda unit: np.where((unit_ids == unit).all(axis=1))[0]
            get_key = lambda unit: tuple(int(x) for x in unit)
        else:
            raise ValueError(
                f"Unsupported shape for unit_ids: {unit_ids.shape}. Only 1D and 2D are supported."
            )

        for unit in unique_units:
            if isinstance(unit, (int, np.integer)):
                key = int(unit)
            else:
                key = get_key(unit)
            indices = get_indices(unit)

            unit_predictions_denorm = predictions[indices]
            unit_targets_denorm = targets[indices]

            # A. Update PRIMARY Metrics (Now store DENORMALIZED results)
            for metric in self.unit_metrics[key].values():
                metric.update(unit_predictions_denorm, unit_targets_denorm)

            # B. Update SECONDARY Metrics (Only if dual reporting is needed)
            if is_denormalized_run:
                unit_predictions_norm = normalized_predictions[indices]
                unit_targets_norm = normalized_targets[indices]

                for metric in self.normalized_unit_metrics[key].values():
                    metric.update(unit_predictions_norm, unit_targets_norm)

    # -------------------------------------------------------------------------
    # 🎯 OVERRIDDEN ARCHITECTURAL METHODS
    # -------------------------------------------------------------------------

    @override
    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        """Extends base saving logic to include unit_ids."""
        # 1. Get standard preds/targets from parent
        dict_to_save = super()._prepare_predictions_for_save()

        if not dict_to_save:
            return dict_to_save

        # 2. Add Unit IDs
        unit_ids = np.concatenate(self.unit_ids, axis=0)

        dict_to_save.update(
            {
                "unit_ids": xr.DataArray(
                    data=unit_ids,
                    dims=["sample"] + [f"unit_dim{i}" for i in range(1, unit_ids.ndim)],
                ),
            }
        )

        return dict_to_save

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        """
        Computes all collected metrics, returning individual per-unit results (if
        log_per_unit_metrics is True) and the global aggregated mean.
        """
        flat_results = {}
        average_collector = defaultdict(list)

        is_denormalized_run = self.scaling_wrapper.apply_inverse

        if not self.unit_metrics:
            # Fallback for empty metrics
            return {f"{name}_mean": 0.0 for name in self.metric_names}

        # --- 1. Compute and Collect Denormalized Results (Primary Metric Set) ---
        for unit_id, metrics in self.unit_metrics.items():
            for name, metric in metrics.items():

                denorm_value = metric.compute()

                # A. Conditionally Log Per-Unit Result (Denormalized)
                if self.log_per_unit_metrics:
                    per_unit_key = f"{name}_{unit_id}_denormalized"
                    flat_results[per_unit_key] = denorm_value

                # B. Collect for Mean Aggregation
                average_collector[f"{name}_denormalized"].append(denorm_value)

        # --- 2. Compute and Collect Normalized Results (Secondary Metric Set) ---
        if is_denormalized_run:
            for unit_id, metrics in self.normalized_unit_metrics.items():
                for name, metric in metrics.items():
                    normalized_value = metric.compute()

                    # C. Conditionally Log Per-Unit Result (Normalized)
                    if self.log_per_unit_metrics:
                        per_unit_key = f"{name}_{unit_id}_normalized"
                        flat_results[per_unit_key] = normalized_value

                    # D. Collect for Mean Aggregation
                    average_collector[f"{name}_normalized"].append(normalized_value)

        # --- 3. Calculate Final Aggregated Means ---
        for name_or_key, values in average_collector.items():
            # Final key examples: 'mse_denormalized_mean', 'mse_normalized_mean'
            flat_results[f"{name_or_key}_mean"] = np.mean(values)

        return flat_results

    @override
    def compute(
        self, mode: str = "val", epoch: int = 0, step: int = 0
    ) -> Dict[str, float]:
        """Orchestrates plotting, saving, and metric computation."""

        if self.log_fit_vs_target_curves:
            if mode != "train" or (
                mode == "val" and epoch % self.log_image_every_n_epochs == 0
            ):
                self._log_pred_vs_target_curves(mode, step)

        return super().compute(mode, epoch, step)

    # -------------------------------------------------------------------------
    # HELPER METHODS (Specific to MultiUnit)
    # -------------------------------------------------------------------------

    def _inverse_transform_per_unit(self, predictions, targets, unit_ids):
        """Applies inverse scaling per unit if unit_ids are available."""
        if unit_ids.ndim not in (1, 2):
            logger.warning(
                f"Unsupported shape for unit_ids: {unit_ids.shape}. Skipping per-unit inverse."
            )
            return self.scaling_wrapper.inverse_transform_if_needed(
                predictions, targets
            )

        if unit_ids.ndim == 1:
            unique_units = np.unique(unit_ids)
            get_indices = lambda unit: np.where(unit_ids == unit)[0]
        else:
            unique_units = np.unique(unit_ids, axis=0)
            get_indices = lambda unit: np.where((unit_ids == unit).all(axis=1))[0]

        pred_shape = predictions.shape
        targets_filled = np.full(targets.shape, -np.inf)
        predictions_filled = np.full(pred_shape, -np.inf)

        for unit in unique_units:
            indices = get_indices(unit)
            if len(indices) == 0:
                continue

            unit_predictions = predictions[indices]
            unit_targets = targets[indices]

            if unit_predictions.size == 0 or unit_targets.size == 0:
                continue

            unit_predictions, unit_targets = (
                self.scaling_wrapper.inverse_transform_if_needed(
                    unit_predictions, unit_targets, metadata={"unit_id": unit}
                )
            )

            predictions_filled[indices] = unit_predictions
            targets_filled[indices] = unit_targets

        if np.isinf(predictions_filled).any() or np.isinf(targets_filled).any():
            if not self.scaling_wrapper.apply_inverse:
                return predictions, targets
            raise ValueError(
                "Some predictions or targets were not filled after inverse transform."
            )

        return predictions_filled, targets_filled

    def _log_pred_vs_target_curves(self, mode: str, step: int) -> None:
        """Log prediction vs target curves for each unit using remote logger."""
        if not self.remote_logger or not self.preds:
            return

        try:
            predictions = np.concatenate(self.preds, axis=0)
            targets = np.concatenate(self.targets, axis=0)
            unit_ids = np.concatenate(self.unit_ids, axis=0)
        except ValueError:
            logger.warning(
                "Could not concatenate predictions/targets (likely empty), skipping plots."
            )
            return

        if unit_ids.ndim == 1:
            unique_units = np.unique(unit_ids)
            get_indices = lambda unit: np.where(unit_ids == unit)[0]
            get_title_str = lambda unit: str(int(unit))
        elif unit_ids.ndim == 2:
            unique_units = np.unique(unit_ids, axis=0)
            get_indices = lambda unit: np.where((unit_ids == unit).all(axis=1))[0]
            get_title_str = lambda unit: str(tuple(int(x) for x in unit))
        else:
            logger.warning(
                f"Unsupported shape for unit_ids: {unit_ids.shape}. Skipping pred_vs_target curves."
            )
            return

        for unit in unique_units:
            indices = get_indices(unit)
            if len(indices) == 0:
                continue

            unit_predictions = predictions[indices]
            unit_targets = targets[indices]

            plot_targets = unit_targets.flatten()
            plot_preds = unit_predictions.flatten()
            title_suffix = ""

            num_points = plot_targets.shape[0]
            if (
                self.enable_plot_subsampling
                and num_points > self.plot_subsample_threshold
                and self.plot_subsample_factor > 1
            ):
                plot_targets = plot_targets[:: self.plot_subsample_factor]
                plot_preds = plot_preds[:: self.plot_subsample_factor]
                title_suffix = f" (Subsampled 1:{self.plot_subsample_factor})"
                logger.debug(
                    f"Subsampling plot for unit {get_title_str(unit)} by factor {self.plot_subsample_factor}. "
                )

            fig, ax = matplotlib.pyplot.subplots()
            ax.plot(plot_targets, label="Target", linestyle="--")
            ax.plot(plot_preds, label="Prediction", alpha=0.8)

            unit_title = get_title_str(unit)
            ax.set_title(f"Unit {unit_title} - Predictions vs Targets{title_suffix}")
            ax.legend()
            ax.grid(True)

            self.log_plot(
                fig=fig,
                title=f"pred_vs_target_unit_{unit_title}",
                mode=mode,
                epoch=0,
                step=step,
            )
            matplotlib.pyplot.close(fig)

    def _plot_reconstructions(
        self, preds: np.ndarray, targets: np.ndarray, mode: str, epoch: int, step: int
    ) -> plt.Figure:
        """Plot 10 reconstructions vs targets at fixed intervals of the testing data"""

        if not self.remote_logger:
            logger.warning(
                "Remote logger is not configured. Skipping reconstruction plots."
            )
            return

        num_samples = len(preds)
        indices = np.linspace(0, num_samples - 1, num=10, dtype=int)

        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()

        num_dims = targets.shape[2]
        colors = plt.get_cmap("tab10").colors

        for i, idx in enumerate(indices):
            ax = axes[i]
            for dim in range(num_dims):
                color = colors[dim % len(colors)]
                ax.plot(
                    targets[idx, :, dim],
                    label=f"Target (Dim {dim})",
                    linestyle="--",
                    color=color,
                )
                ax.plot(
                    preds[idx, :, dim], label=f"Prediction (Dim {dim})", color=color
                )
                ax.set_title(f"Sample {idx}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid()
        fig.suptitle(f"Reconstruction (Split: {mode}, Epoch: {epoch}, Step: {step})")

        plt.tight_layout()
        return fig
