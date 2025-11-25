"""
Transformer-specific evaluator implementation for time series forecasting.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, override
from pathlib import Path
from collections import defaultdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Assuming these imports are correct for your project structure:
from lmetk.metrics.metric_factory import MetricFactory
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.metrics.base import AbstractMetric
from .base import AbstractEvaluator  # Assuming AbstractEvaluator is in .base

from ..baselines.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS, FORECASTING_TASKS

logger = logging.getLogger(__name__)


class DefaultEvaluator(AbstractEvaluator):
    """Unified evaluator for regression and classification with dual metric reporting."""

    def __init__(
        self,
        metric_names: List[str],
        task_type: str = "regression",
        num_classes: Optional[int] = None,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[str] = None,
        remote_logger: Optional[Any] = None,
        collect_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            paths=paths,
            **kwargs,
        )

        self.task_type = task_type.lower()
        assert CLASSIFICATION_TASKS, f"Unsupported task_type: {task_type}"
        assert REGRESSION_TASKS, f"Unsupported task_type: {task_type}"
        assert FORECASTING_TASKS, f"Unsupported task_type: {task_type}"

        self.metrics = {}  # Primary metrics (will store DENORMALIZED if scaling is ON)
        self.paths = paths
        self.save_predictions = save_predictions
        self.out_path = None
        self.remote_logger = remote_logger
        self.collect_predictions = save_predictions or collect_predictions
        self.metric_names = [name.lower() for name in metric_names]
        self.num_classes = num_classes

        if self.collect_predictions:
            self.preds, self.targets = [], []
            self.scaled_preds, self.scaled_targets = [], []

        # Helper to create metrics
        def _create_metric(name):
            if self.task_type in CLASSIFICATION_TASKS:
                return MetricFactory.create_classification_metric(
                    name, num_classes=num_classes
                )
            else:
                return MetricFactory.create_metric(name, paths=paths)

        # Initialize Primary Metrics
        for name in self.metric_names:
            self.metrics[name] = _create_metric(name)

        # Initialize Secondary Metrics (Only if dual reporting is needed)
        if self.scaling_wrapper.apply_inverse:
            self.normalized_metrics = {}
            for name in self.metric_names:
                self.normalized_metrics[name] = _create_metric(name)

    @override
    def reset(self):
        if self.collect_predictions:
            self.preds.clear()
            self.targets.clear()
            self.scaled_preds.clear()
            self.scaled_targets.clear()

        # Reset Primary Metrics
        for metric in self.metrics.values():
            metric.reset()

        # Reset Secondary Metrics
        if hasattr(self, "normalized_metrics"):
            for metric in self.normalized_metrics.values():
                metric.reset()

    @override
    def update(self, model_out: dict):
        """
        Update metrics, performing dual metric updates for normalized and
        denormalized data when scaling is active.
        Update metrics with new predictions and targets.
        Regression: preds (N, T=1, 1), targets (N, T=1, 1)
        Classification: preds (N, T, C), targets (N, T, 1)
        Forecasting: preds (N, T, 1), targets (N, T, 1)
        Note: At the moment, only single-target tasks are supported!
        """
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        assert targets.ndim == 3 or targets.shape[2] == 1, f"Expected shape (N, T, 1), got {targets.shape}"
        is_denormalized_run = self.scaling_wrapper.apply_inverse

        if self.task_type in CLASSIFICATION_TASKS:
            assert self.scaling_wrapper.inverse_transform is None, "Scaling must be None for classification"
            assert predictions.ndim == 3, f"Expected shape (N, T, {self.num_classes}), got {targets.shape}"
            assert predictions.shape[2] == self.num_classes, f"Expected shape (N, T, {self.num_classes}), got {targets.shape}"
            targets = targets.astype(int)  # ensure integer labels

        elif self.task_type in REGRESSION_TASKS:
            assert predictions.ndim == 3, f"Expected shape (N, T, 1), got {targets.shape}"
            predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
                predictions, targets
            )

        elif self.task_type in FORECASTING_TASKS:
            assert predictions.ndim == 3, f"Expected shape (N, T, 1), got {targets.shape}"
            predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
                predictions, targets
            )

        else:
            raise ValueError(f"Evaluation does not support task_type: {self.task_type}")

        # 1. Store Normalized Data (BEFORE inverse transform)
        if is_denormalized_run:
            # We need a copy of the normalized data for the secondary metric set
            normalized_predictions = predictions.copy()
            normalized_targets = targets.copy()

            # Store original data for file saving later (self.preds/self.targets holds normalized)
            if self.collect_predictions:
                self.preds.append(predictions.copy())
                self.targets.append(targets.copy())

        elif self.collect_predictions:
            # Store original data for file saving when scaling is OFF
            self.preds.append(predictions)
            self.targets.append(targets)

        # 2. Apply inverse scaling (MODIFIES predictions/targets)
        predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
            predictions, targets
        )

        # 3. Store Denormalized Data
        if self.collect_predictions and is_denormalized_run:
            self.scaled_preds.append(predictions)
            self.scaled_targets.append(targets)

        # 4. Metric Update (DUAL UPDATE LOGIC)

        # 4a. Update PRIMARY Metrics (Will store DENORMALIZED results)
        for metric in self.metrics.values():
            metric.update(predictions, targets)

        # 4b. Update SECONDARY Metrics (Will store NORMALIZED results)
        if is_denormalized_run:
            for metric in self.normalized_metrics.values():
                metric.update(normalized_predictions, normalized_targets)

    # FIX: Corrected indentation to class level
    def log_plot(self, fig: plt.Figure, title: str, mode: str, epoch: int, step: int):
        if self.remote_logger:
            self.remote_logger.experiment.log({f"plots/{mode}/{title}": fig})

    def _prepare_predictions_for_save(self) -> Dict[str, Any]:
        """
        Concatenates stored predictions/targets and prepares the dictionary
        for saving using xarray.Dataset.
        """
        dict_to_save = {}

        if not self.save_predictions or not self.collect_predictions:
            return dict_to_save

        is_denormalized_run = self.scaling_wrapper.apply_inverse

        if is_denormalized_run:
            # If scaling is ON: Primary save uses DENORMALIZED data
            preds = np.concatenate(self.scaled_preds, axis=0)
            targets = np.concatenate(self.scaled_targets, axis=0)

            # Use self.preds/self.targets for the normalized data
            normalized_preds = np.concatenate(self.preds, axis=0)
            normalized_targets = np.concatenate(self.targets, axis=0)
        else:
            # If scaling is OFF: Primary save uses NORMALIZED data
            preds = np.concatenate(self.preds, axis=0)
            targets = np.concatenate(self.targets, axis=0)

        if self.task_type in CLASSIFICATION_TASKS and preds.shape[-1] > 1:
            preds = np.argmax(preds, axis=-1, keepdims=True)

        # Determine dimension names for xarray
        d_names = ["sample"] + [f"dim{i}" for i in range(1, preds.ndim)]

        # 1. Save PRIMARY (Denormalized or Original) predictions
        dict_to_save.update({"preds": (d_names, preds), "targets": (d_names, targets)})

        # 2. Save SECONDARY (Normalized) predictions, if applicable
        if is_denormalized_run:
            dict_to_save.update(
                {
                    "preds_normalized": (
                        d_names,
                        normalized_preds,
                    ),
                    "targets_normalized": (
                        d_names,
                        normalized_targets,
                    ),
                }
            )

        return dict_to_save

    def _save_predictions_to_file(self, dict_to_save: Dict[str, Any], mode: str):
        """
        Handles path setup and saves the prepared prediction data dictionary
        to a netCDF file.
        """
        if not dict_to_save:
            return

        if self.out_path is None:
            if self.paths and getattr(self.paths, "eval_details", None):
                self.out_path = Path(self.paths.eval_details) / mode
                self.out_path.mkdir(parents=True, exist_ok=True)
            assert (
                self.out_path is not None
            ), "Output path must be specified to save predictions."

        try:
            df = xr.Dataset(dict_to_save)
            df.to_netcdf(self.out_path / "predictions.nc")
            logger.info(f"Predictions saved to {self.out_path / 'predictions.nc'}")
        except Exception as e:
            logger.error(
                f"Failed to save predictions to netCDF at {self.out_path}: {e}"
            )

    @override
    def _compute_metrics(self) -> Dict[str, float]:
        """
        Computes all collected metrics, including both normalized and denormalized results.
        """
        results = {}
        is_denormalized_run = self.scaling_wrapper.apply_inverse

        # 1. Compute Primary Metrics (Denormalized or Original)
        for name, metric in self.metrics.items():
            results[f"{name}_denormalized"] = metric.compute()

        # 2. Compute Secondary Metrics (Normalized)
        if is_denormalized_run:
            for name, metric in self.normalized_metrics.items():
                results[f"{name}_normalized"] = metric.compute()
        else:
            # FIX: If scaling is OFF, the primary metric is actually the normalized metric.
            # We must rename the primary keys to reflect they are the normalized version.
            results_renamed = {}
            for key, value in results.items():
                results_renamed[key.replace("_denormalized", "_normalized")] = value
            results = results_renamed

        return results

    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Orchestrates the computation, saving, and reporting steps.
        """
        # 1. Prepare Data for Saving (Calls overridable method)
        dict_to_save = self._prepare_predictions_for_save()

        # 2. Handle Output Path and Saving (Isolated logic)
        self._save_predictions_to_file(dict_to_save, mode)

        # 3. Compute and Return Metrics (Isolated logic)
        return self._compute_metrics()


# class ClassificationEvaluator(DefaultEvaluator):
#     """Evaluator for classification tasks."""

#     def __init__(
#         self,
#         num_classes: int,
#         metric_names: List[str],
#         inverse_transform: Optional[InverseTransformMixin] = None,
#         apply_inverse_scaling: bool = False,
#         save_predictions: bool = False,
#         paths: Optional[str] = None,
#         **kwargs,
#     ):
#         """
#         Initialize the evaluator with specified metrics and optional scaling.

#         Parameters
#         ----------
#         metric_names : List[str]
#             List of metric names to compute.
#         inverse_transform : Optional[Any]
#             The scaler object with inverse_transform method
#         apply_inverse : bool
#             Whether to apply inverse scaling before computing metrics
#         """
#         super().__init__(
#             metric_names=metric_names,
#             num_classes=num_classes,
#             task_type="classification",
#             inverse_transform=inverse_transform,
#             apply_inverse_scaling=apply_inverse_scaling,
#             save_predictions=save_predictions,
#             paths=paths,
#             **kwargs,
#         )

#     @override
#     def _prepare_predictions_for_save(self) -> Dict[str, Any]:
#         """
#         Custom saving logic for classification: saves logits as predictions
#         and ensures targets are properly formatted (B, T, C).
#         """
#         dict_to_save = {}

#         if not self.save_predictions or not self.collect_predictions:
#             return dict_to_save

#         # Predictions are logits (B, T, L)
#         preds = np.concatenate(self.preds, axis=0)

#         # Targets are integer labels (B, T, 1) and must be converted to one-hot (B, T, C)
#         # Note: Targets are integer labels (B, T, 1) when collected in update(), but here
#         # we treat them as needing to match the pred dimension for consistency (B,T,L)

#         # Assuming targets are already (B, T, C) or (B, T, 1) and we need to match pred dims
#         targets = np.concatenate(self.targets, axis=0)

#         # Check if targets are 1D/2D and need an extra class dimension (C) for logging,
#         # although usually classification targets are saved as integers.
#         # Given your note: "preds are logits with dims (B,T,L) where L is logits
#         # and (B,T,C) for targets where C is classes," we'll assume the targets
#         # list contains the final, 3D target shape (B, T, C) or (B, T, L) for safety.

#         # Determine dimension names for xarray
#         # Use the dimensions of the predictions (logits)
#         d_names = ["sample"] + ["time"] + ["class_logits"]

#         # Save Logits
#         dict_to_save.update({"preds_logits": (d_names, preds)})

#         # Determine target dimension names (assuming they match prediction dimensions L=C for save)
#         t_names = ["sample"] + ["class_labels"]

#         # Save Targets (as collected, usually integer labels)
#         dict_to_save.update({"targets_labels": (t_names, targets)})

#         # Skip denormalized saving as scaling is usually not applied to logits/integer labels

#         return dict_to_save

#     @override
#     def _compute_metrics(self) -> Dict[str, float]:
#         """
#         Computes all collected metrics, then removes '_normalized' and
#         '_denormalized' suffixes, as they are nonsensical for classification.
#         """
#         # 1. Run the base computation logic
#         # This returns keys like: {'accuracy_normalized', 'precision_normalized'}
#         # OR (if scaling was on): {'accuracy_denormalized', 'accuracy_normalized'}
#         results = super()._compute_metrics()

#         # 2. Filter and Rename Metrics
#         final_results = {}

#         for key, value in results.items():
#             # Get the base metric name (e.g., 'accuracy', 'f1')

#             # Remove suffixes: '_normalized', '_denormalized'

            # Now, remove the task-irrelevant suffixes:
            if key.endswith("_normalized"):
                base_key = key.removesuffix("_normalized")
            if key.endswith("_denormalized"):
                base_key = key.removesuffix("_denormalized")

#             # Final key is just the base name (e.g., 'accuracy')
#             # If a base metric appears twice (e.g., normalized and denormalized),
#             # we must decide which one to keep, or if both should be kept under
#             # different names (if scaling was actually applied).

#             # Since denormalized results are nonsensical in CLASSIFICATION,
#             # we prioritize the BASE result which implicitly uses the final labels.

            # We will use the base name as the final key

            if base_key in final_results:
                assert (
                    final_results[base_key] == value
                ), f"Conflict in metric values for {base_key}: {final_results[base_key]} vs {value}"
            else:
                final_results[base_key] = value

#         return final_results


# class ForecastingEvaluator(DefaultEvaluator):

#     def __init__(
#         self,
#         target_dim_position: int,
#         metric_names: List[str],
#         model_seq_len: int,
#         model_label_len: int,
#         effective_pred_len: Optional[int] = None,
#         model_pred_len: Optional[int] = None,
#         inverse_transform: Optional[InverseTransformMixin] = None,
#         apply_inverse_scaling: bool = False,
#         save_predictions: bool = False,
#         paths: Optional[str] = None,
#         **kwargs,
#     ):
#         """
#         Initialize the evaluator with specified metrics and optional scaling.

#         Parameters
#         ----------
#         metric_names : List[str]
#             List of metric names to compute.
#         inverse_transform : Optional[Any]
#             The scaler object with inverse_transform method
#         apply_inverse : bool
#             Whether to apply inverse scaling before computing metrics
#         """
#         super().__init__(
#             metric_names=metric_names,
#             inverse_transform=inverse_transform,
#             apply_inverse_scaling=apply_inverse_scaling,
#             save_predictions=save_predictions,
#             paths=paths,
#             **kwargs,
#         )

#         self.model_label_len = model_label_len
#         self.model_pred_len = model_pred_len
#         self.model_seq_len = model_seq_len
#         self.effective_pred_len = effective_pred_len

#         self.target_dim_position = target_dim_position

#         if self.model_pred_len is None:
#             raise ValueError("Model's pred_len (model_pred_len) must be set.")

#         if (
#             self.effective_pred_len is not None
#             and self.model_pred_len < self.effective_pred_len
#         ):
#             raise ValueError(
#                 f"effective_pred_len ({self.effective_pred_len}) cannot be greater than "
#                 f"model_pred_len ({self.model_pred_len})."
#             )

#     @override
#     def update(self, model_out: dict):

#         # Batch, Seq. Lenght, Feature Dim
#         predictions = model_out["predictions"]
#         targets = model_out["targets"]

#         # TODO: R. do you agree with this logic?
#         if self.effective_pred_len is not None:
#             predictions = predictions[:, -self.effective_pred_len :, :]
#             targets = targets[:, -self.effective_pred_len :, :]

#         if predictions.shape != targets.shape:
#             raise ValueError(
#                 f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
#             )

#         # Based on the feature mode take the features:
#         # f_dim = 1 if self.task_mode == 'MS' else: None

#         # TODO: logic here need to be improved as self.target_dim_position is not always defined as None
#         # running etth and railway datasets here is important checks

#         # if self.task_mode == 'univariate' or
#         # (self.task_mode == 'multivariate' and self.target_dim_position
#         # is not None):
#         if self.task_mode == "univariate" and self.target_dim_position is not None:
#             f_dim = self.target_dim_position
#             predictions, targets = predictions[:, :, f_dim], targets[:, :, f_dim]

#         model_out["predictions"] = predictions
#         model_out["targets"] = targets
#         super().update(model_out)


# class ReconstructionEvaluator(DefaultEvaluator):
#     """Evaluator for reconstruction tasks."""

#     def __init__(
#         self,
#         metric_names: List[str],
#         inverse_transform: Optional[InverseTransformMixin] = None,
#         apply_inverse_scaling: bool = False,
#         save_predictions: bool = False,
#         plot_reconstructions: bool = False,
#         paths: Optional[str] = None,
#         **kwargs,
#     ):
#         """
#         Initialize the evaluator with specified metrics and optional scaling.

#         Parameters
#         ----------
#         metric_names : List[str]
#             List of metric names to compute.
#         inverse_transform : Optional[Any]
#             The scaler object with inverse_transform method
#         apply_inverse : bool
#             Whether to apply inverse scaling before computing metrics
#         """
#         collect_predictions = plot_reconstructions

#         matplotlib.use("Agg")  # Use a non-interactive backend
#         self.plot_reconstructions = plot_reconstructions
#         self.plots_out_path = None

#         super().__init__(
#             metric_names=metric_names,
#             inverse_transform=inverse_transform,
#             apply_inverse_scaling=apply_inverse_scaling,
#             save_predictions=save_predictions,
#             collect_predictions=collect_predictions,
#             paths=paths,
#             **kwargs,
#         )

#     def _plot_reconstructions(
#         self, preds: np.ndarray, targets: np.ndarray, mode: str, epoch: int, step: int
#     ) -> plt.Figure:
#         """Plot 10 reconstructions vs targets at fixed intervals of the testing data"""

#         if not self.remote_logger:
#             logger.warning(
#                 "Remote logger is not configured. Skipping reconstruction plots."
#             )
#             return

#         # Select 10 evenly spaced indices for plotting
#         num_samples = len(preds)
#         indices = np.linspace(0, num_samples - 1, num=10, dtype=int)

#         fig, axes = plt.subplots(
#             5, 2, figsize=(15, 20)
#         )  # Create a 5x2 grid of subplots
#         axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

#         # Assuming targets and preds have shape (samples, time, dims)
#         num_dims = targets.shape[2]
#         colors = plt.get_cmap("tab10").colors  # Use a colormap for distinct colors

#         for i, idx in enumerate(indices):
#             ax = axes[i]
#             for dim in range(num_dims):
#                 color = colors[dim % len(colors)]  # Cycle through colors if dims > 10
#                 ax.plot(
#                     targets[idx, :, dim],
#                     label=f"Target (Dim {dim})",
#                     linestyle="--",
#                     color=color,
#                 )
#                 ax.plot(
#                     preds[idx, :, dim], label=f"Prediction (Dim {dim})", color=color
#                 )
#                 ax.set_title(f"Sample {idx}")
#                 ax.set_xlabel("Time")
#                 ax.set_ylabel("Value")
#                 ax.legend()
#                 ax.grid()
#         fig.suptitle(f"Reconstruction (Split: {mode}, Epoch: {epoch}, Step: {step})")

#         plt.tight_layout()  # Adjust layout to prevent overlap
#         return fig

#     def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:

#         if self.plots_out_path is None:
#             if self.paths and getattr(self.paths, "plot_dir", None):
#                 self.plots_out_path = Path(self.paths.plot_dir) / mode
#                 self.plots_out_path.mkdir(parents=True, exist_ok=True)

#             assert (
#                 self.plots_out_path is not None
#             ), "Output path must be specified to save predictions."

#         if self.plot_reconstructions:
#             preds = np.concatenate(self.preds, axis=0)
#             targets = np.concatenate(self.targets, axis=0)
#             fig_recons = self._plot_reconstructions(preds, targets, mode, epoch, step)
#             if fig_recons:
#                 self.log_plot(fig_recons, "reconstructions_overview", mode, epoch, step)
#                 fig_recons.savefig(
#                     self.plots_out_path / f"reconstructions_overview_{epoch}_{step}.png"
#                 )
#                 plt.close(fig_recons)

#         return super().compute(mode, epoch=epoch, step=step)
