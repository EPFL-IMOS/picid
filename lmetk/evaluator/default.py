"""
Transformer-specific evaluator implementation for time series forecasting.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, override

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from lmetk.metrics.metric_factory import MetricFactory
from lmetk.transforms.base.multisource import InverseTransformMixin

from .base import AbstractEvaluator
import matplotlib

logger = logging.getLogger(__name__)


class DefaultEvaluator(AbstractEvaluator):
    """Unified evaluator for regression and classification."""

    def __init__(
        self,
        metric_names: List[str],
        task_type: str = "regression",  # "regression" or "classification"
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
        assert self.task_type in [
            "regression",
            "classification",
        ], f"Unsupported task_type: {task_type}"

        self.metrics = {}
        self.paths = paths
        self.save_predictions = save_predictions
        self.out_path = None
        self.remote_logger = remote_logger
        self.collect_predictions = save_predictions or collect_predictions

        if self.collect_predictions:
            self.preds, self.targets = [], []
            self.scaled_preds, self.scaled_targets = [], []

        for name in metric_names:
            if self.task_type == "classification":
                metric = MetricFactory.create_classification_metric(
                    name, num_classes=num_classes
                )
            else:
                metric = MetricFactory.create_metric(name, paths=paths)
            self.metrics[name.lower()] = metric
        self.metric_names = [name.lower() for name in metric_names]

    @override
    def reset(self):
        if self.collect_predictions:
            self.preds.clear()
            self.targets.clear()
            self.scaled_preds.clear()
            self.scaled_targets.clear()
        for metric in self.metrics.values():
            metric.reset()

    @override
    def update(self, model_out: dict):
        """
        Update metrics with new predictions and targets.
        Regression: (N, T, D)
        Classification: preds (N, T, C), targets (N, T, 1)
        """
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        if self.task_type == "classification":
            targets = targets.astype(int)  # ensure integer labels

        if self.collect_predictions:
            self.preds.append(predictions)
            self.targets.append(targets)

        # Apply inverse scaling if needed (rare for classification)
        predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
            predictions, targets
        )

        if self.collect_predictions and self.scaling_wrapper.apply_inverse:
            self.scaled_preds.append(predictions)
            self.scaled_targets.append(targets)

        # Update metrics
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    def log_plot(self, fig: plt.Figure, title: str, mode: str, epoch: int, step: int):
        if self.remote_logger:
            self.remote_logger.experiment.log({f"plots/{mode}/{title}": fig})

    @override
    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:
        """
        Compute final metric values.

        Returns
        -------
        Dict[str, float]
            Dictionary with metric names as keys and computed values as values
        """
        dict_to_save = {}

        if self.save_predictions:
            preds = np.concatenate(self.preds, axis=0)
            targets = np.concatenate(self.targets, axis=0)
            d_names = ["sample"] + [f"dim{i}" for i in range(1, preds.ndim)]

            dict_to_save.update(
                {"preds": (d_names, preds), "targets": (d_names, targets)}
            )
        elif self.save_predictions and self.scaling_wrapper.apply_inverse:
            dict_to_save.update(
                {
                    "preds_denormalized": (
                        d_names,
                        np.concatenate(self.scaled_preds, axis=0),
                    ),
                    "targets_denormalized": (
                        d_names,
                        np.concatenate(self.scaled_targets, axis=0),
                    ),
                }
            )

        if self.out_path is None:
            if self.paths and getattr(self.paths, "eval_details", None):
                self.out_path = Path(self.paths.eval_details) / mode
                self.out_path.mkdir(parents=True, exist_ok=True)
            assert (
                self.out_path is not None
            ), "Output path must be specified to save predictions."

        if len(dict_to_save) > 0:
            df = xr.Dataset(dict_to_save)
            df.to_netcdf(self.out_path / "predictions.nc")

        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()

        return results


class ClassificationEvaluator(DefaultEvaluator):
    """Evaluator for classification tasks."""

    def __init__(
        self,
        num_classes: int,
        metric_names: List[str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator with specified metrics and optional scaling.

        Parameters
        ----------
        metric_names : List[str]
            List of metric names to compute.
        inverse_transform : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling before computing metrics
        """
        super().__init__(
            metric_names=metric_names,
            num_classes=num_classes,
            task_type="classification",
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            paths=paths,
            **kwargs,
        )


class ForecastingEvaluator(DefaultEvaluator):

    def __init__(
        self,
        target_dim_position: int,
        metric_names: List[str],
        model_seq_len: int,
        model_label_len: int,
        effective_pred_len: Optional[int] = None,
        model_pred_len: Optional[int] = None,
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator with specified metrics and optional scaling.

        Parameters
        ----------
        metric_names : List[str]
            List of metric names to compute.
        inverse_transform : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling before computing metrics
        """
        super().__init__(
            metric_names=metric_names,
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            paths=paths,
            **kwargs,
        )

        self.model_label_len = model_label_len
        self.model_pred_len = model_pred_len
        self.model_seq_len = model_seq_len
        self.effective_pred_len = effective_pred_len

        self.target_dim_position = target_dim_position

        if self.model_pred_len is None:
            raise ValueError("Model's pred_len (model_pred_len) must be set.")

        if (
            self.effective_pred_len is not None
            and self.model_pred_len < self.effective_pred_len
        ):
            raise ValueError(
                f"effective_pred_len ({self.effective_pred_len}) cannot be greater than "
                f"model_pred_len ({self.model_pred_len})."
            )

    @override
    def update(self, model_out: dict):

        # Batch, Seq. Lenght, Feature Dim
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        # TODO: R. do you agree with this logic?
        if self.effective_pred_len is not None:
            predictions = predictions[:, -self.effective_pred_len :, :]
            targets = targets[:, -self.effective_pred_len :, :]

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # Based on the feature mode take the features:
        # f_dim = 1 if self.task_mode == 'MS' else: None

        # TODO: logic here need to be improved as self.target_dim_position is not always defined as None
        # running etth and railway datasets here is important checks

        # if self.task_mode == 'univariate' or
        # (self.task_mode == 'multivariate' and self.target_dim_position
        # is not None):
        if self.task_mode == "univariate" and self.target_dim_position is not None:
            f_dim = self.target_dim_position
            predictions, targets = predictions[:, :, f_dim], targets[:, :, f_dim]

        model_out["predictions"] = predictions
        model_out["targets"] = targets
        super().update(model_out)


class ReconstructionEvaluator(DefaultEvaluator):
    """Evaluator for reconstruction tasks."""

    def __init__(
        self,
        metric_names: List[str],
        inverse_transform: Optional[InverseTransformMixin] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        plot_reconstructions: bool = False,
        paths: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the evaluator with specified metrics and optional scaling.

        Parameters
        ----------
        metric_names : List[str]
            List of metric names to compute.
        inverse_transform : Optional[Any]
            The scaler object with inverse_transform method
        apply_inverse : bool
            Whether to apply inverse scaling before computing metrics
        """
        collect_predictions = plot_reconstructions

        matplotlib.use("Agg")  # Use a non-interactive backend
        self.plot_reconstructions = plot_reconstructions
        self.plots_out_path = None

        super().__init__(
            metric_names=metric_names,
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            collect_predictions=collect_predictions,
            paths=paths,
            **kwargs,
        )

    def _plot_reconstructions(
        self, preds: np.ndarray, targets: np.ndarray, mode: str, epoch: int, step: int
    ) -> plt.Figure:
        """Plot 10 reconstructions vs targets at fixed intervals of the testing data"""

        if not self.remote_logger:
            logger.warning(
                "Remote logger is not configured. Skipping reconstruction plots."
            )
            return

        # Select 10 evenly spaced indices for plotting
        num_samples = len(preds)
        indices = np.linspace(0, num_samples - 1, num=10, dtype=int)

        fig, axes = plt.subplots(
            5, 2, figsize=(15, 20)
        )  # Create a 5x2 grid of subplots
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        # Assuming targets and preds have shape (samples, time, dims)
        num_dims = targets.shape[2]
        colors = plt.get_cmap("tab10").colors  # Use a colormap for distinct colors

        for i, idx in enumerate(indices):
            ax = axes[i]
            for dim in range(num_dims):
                color = colors[dim % len(colors)]  # Cycle through colors if dims > 10
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

        plt.tight_layout()  # Adjust layout to prevent overlap
        return fig

    def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]:

        if self.plots_out_path is None:
            if self.paths and getattr(self.paths, "plot_dir", None):
                self.plots_out_path = Path(self.paths.plot_dir) / mode
                self.plots_out_path.mkdir(parents=True, exist_ok=True)

            assert (
                self.plots_out_path is not None
            ), "Output path must be specified to save predictions."

        if self.plot_reconstructions:
            preds = np.concatenate(self.preds, axis=0)
            targets = np.concatenate(self.targets, axis=0)
            fig_recons = self._plot_reconstructions(preds, targets, mode, epoch, step)
            if fig_recons:
                self.log_plot(fig_recons, "reconstructions_overview", mode, epoch, step)
                fig_recons.savefig(
                    self.plots_out_path / f"reconstructions_overview_{epoch}_{step}.png"
                )
                plt.close(fig_recons)

        return super().compute(mode, epoch=epoch, step=step)
