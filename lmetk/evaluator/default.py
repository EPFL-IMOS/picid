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
        assert self.task_type in ALL_TASKS, f"Unsupported task_type: {self.task_type}"

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

        is_denormalized_run = self.scaling_wrapper.apply_inverse

        assert (
            targets.ndim == 3 and targets.shape[2] == 1
        ), f"Expected target shape (N, T, 1), got {targets.shape}"

        if self.task_type in CLASSIFICATION_TASKS:
            assert (
                self.scaling_wrapper.inverse_transform is None
            ), "Scaling must be None for classification"
            assert (
                predictions.ndim == 3 and predictions.shape[2] == self.num_classes
            ), f"Expected prediction shape (N, T, {self.num_classes}), got {predictions.shape}"
            targets = targets.astype(int)  # ensure integer labels

        elif self.task_type in REGRESSION_TASKS:
            assert (
                predictions.ndim == 3 and predictions.shape[2] == 1
            ), f"Expected prediction shape (N, T, 1), got {predictions.shape}"
            predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
                predictions, targets
            )

        elif self.task_type in FORECASTING_TASKS:
            assert (
                predictions.ndim == 3 and predictions.shape[2] == 1
            ), f"Expected prediction shape (N, T, 1), got {predictions.shape}"
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

        df = xr.Dataset(dict_to_save)
        df.to_netcdf(self.out_path / "predictions.nc")
        logger.info(f"Predictions saved to {self.out_path / 'predictions.nc'}")

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
