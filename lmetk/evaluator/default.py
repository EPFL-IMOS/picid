"""
Transformer-specific evaluator implementation for time series forecasting.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, override

import numpy as np
import xarray as xr

from lmetk.metrics.metric_factory import MetricFactory
from lmetk.transforms.base.multisource import InverseTransformMixin

from .base import AbstractEvaluator


class DefaultEvaluator(AbstractEvaluator):
    """Default evaluator that updates metrics incrementally."""

    """Sequential evaluator that updates metrics incrementally."""

    def __init__(
        self,
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
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            paths=paths,
            **kwargs,
        )

        # Validate and create metrics
        self.metrics = {}
        self.paths = paths
        self.save_predictions = save_predictions
        self.out_path = None

        if save_predictions:
            self.preds = []
            self.targets = []
            self.scaled_preds = []
            self.scaled_targets = []

        for name in metric_names:
            metric = MetricFactory.create_metric(name, paths=paths)
            self.metrics[name.lower()] = metric

        self.metric_names = [name.lower() for name in metric_names]

    @override
    def reset(self):
        """Reset all metrics."""
        if self.save_predictions:
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

        Parameters
        ----------
        model_out : dict
            Dictionary containing 'predictions' and 'targets' keys with numpy arrays
        """
        # Batch, Seq. Lenght, Feature Dim
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        if self.save_predictions:
            self.preds.append(predictions)
            self.targets.append(targets)

        # Apply inverse scaling if configured
        predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
            predictions, targets
        )

        # Save predictions and targets if configured
        if self.save_predictions and self.scaling_wrapper.apply_inverse:
            self.scaled_preds.append(predictions)
            self.scaled_targets.append(targets)

        # Update all metrics
        for metric in self.metrics.values():
            metric.update(predictions, targets)

    @override
    def compute(self, mode: str) -> Dict[str, float]:
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
                {
                    "preds": (d_names, preds),
                    "targets": (d_names, targets),
                }
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
