"""
Multiunit evaluator implementation.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

import logging

from typing import Any, Dict, List, Optional, override
import numpy as np
from pathlib import Path
import xarray as xr

from lmetk.metrics.metric_factory import MetricFactory
from collections import defaultdict
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.metrics.base import AbstractMetric

from lmetk.evaluator.base import AbstractEvaluator
import matplotlib

logger = logging.getLogger(__name__)


class MultiUnitEvaluator(AbstractEvaluator):
    """
    Evaluator that handles per-unit metrics, inverse scaling, and prediction saving.
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
        **kwargs,
    ):
        super().__init__(
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            paths=paths,
            **kwargs,
        )

        self.metric_names = [name.lower() for name in metric_names]
        self.apply_inverse_scaling = apply_inverse_scaling
        self.scaler = inverse_transform

        # --- Logic from DefaultEvaluator ---
        self.out_path = None
        self.paths = paths
        self.save_predictions = save_predictions
        self.remote_logger = remote_logger
        self.collect_predictions = save_predictions or collect_predictions
        self.reset_collections()  # Initialize prediction lists

        # --- Logic from MultiUnitEvaluator ---
        # This will hold the metric objects for each unit.
        # e.g., { unit_1: {'mpe': MPE_Metric()}, unit_2: {'mpe': MPE_Metric()} }
        self.unit_metrics = defaultdict(self._create_metric_dict)
        self.reset()

    def _create_metric_dict(self) -> Dict[str, AbstractMetric]:
        """Factory function to create a dictionary of new metric instances."""
        return {
            name: MetricFactory.create_metric(name, self.paths)
            for name in self.metric_names
        }

    def reset_collections(self):
        if self.collect_predictions:
            self.preds = []
            self.targets = []
            self.scaled_preds = []
            self.scaled_targets = []
            self.unit_ids = []

    @override
    def reset(self):
        """Reset all metrics and collected predictions."""
        self.reset_collections()
        self.unit_metrics.clear()

    @override
    def update(self, model_out: dict):
        """Update metrics with new predictions, handling scaling and per-unit logic."""
        predictions = model_out["predictions"]
        targets = model_out["targets"]
        unit_ids = model_out["unit_id"]

        if self.collect_predictions:
            self.preds.append(predictions)
            self.targets.append(targets)
            self.unit_ids.append(unit_ids)

        # Apply inverse scaling if configured
        predictions, targets = self.scaling_wrapper.inverse_transform_if_needed(
            predictions, targets
        )

        # Save predictions and targets if configured
        if self.collect_predictions and self.scaling_wrapper.apply_inverse:
            self.scaled_preds.append(predictions)
            self.scaled_targets.append(targets)

        # --- Per-Unit Update Logic ---
        # Step 1: Determine how to find unique units and their keys based on shape.
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

        # Step 2: Loop through unique units and update metrics using a unified logic.
        for unit in unique_units:
            key = get_key(unit)
            indices = get_indices(unit)

            unit_predictions = predictions[indices]
            unit_targets = targets[indices]

            # Update all metrics associated with this unit key
            for metric in self.unit_metrics[key].values():
                metric.update(unit_predictions, unit_targets)

    @override
    def compute(
        self, mode: str = "val", epoch: int = 0, step: int = 0
    ) -> Dict[str, float]:
        """Compute final metrics, save predictions, and return flattened results."""
        # --- Saving logic extracted into helper ---
        self._save_predictions_if_needed(mode)

        # --- Metric computation logic from MultiUnitEvaluator ---
        flat_results = {}
        average_collector = defaultdict(list)

        if not self.unit_metrics:
            return {f"{name}_mean": 0.0 for name in self.metric_names}

        # !!! Uncomment to get the per-unit metrics logged
        # for unit_id, metrics in self.unit_metrics.items():
        #     for name, metric in metrics.items():
        #         value = metric.compute()
        #         flat_results[f"{name}_{unit_id}"] = value
        #         average_collector[name].append(value)

        for name, values in average_collector.items():
            flat_results[f"{name}_mean"] = np.mean(values)

        for name, values in average_collector.items():
            flat_results[f"{name}_std"] = np.std(values)

        return flat_results

    def _save_predictions_if_needed(self, mode: str) -> None:
        """Save collected predictions/targets to disk when configured."""
        dict_to_save = {}

        if self.save_predictions and self.collect_predictions:
            predictions = np.concatenate(self.preds, axis=0)
            targets = np.concatenate(self.targets, axis=0)
            unit_ids = np.concatenate(self.unit_ids, axis=0)

            d_names = ["sample"] + [f"dim{i}" for i in range(1, predictions.ndim)]
            dict_to_save.update(
                {
                    "predictions": xr.DataArray(
                        data=predictions,
                        dims=d_names,
                    ),
                    "targets": xr.DataArray(
                        data=targets,
                        dims=d_names,
                    ),
                    "unit_ids": xr.DataArray(
                        data=unit_ids,
                        dims=["sample"]
                        + [f"unit_dim{i}" for i in range(1, unit_ids.ndim)],
                    ),
                }
            )

            if self.apply_inverse_scaling:
                scaled_predictions = np.concatenate(self.scaled_preds, axis=0)
                scaled_targets = np.concatenate(self.scaled_targets, axis=0)

                dict_to_save.update(
                    {
                        "scaled_predictions": xr.DataArray(
                            data=scaled_predictions,
                            dims=d_names,
                        ),
                        "scaled_targets": xr.DataArray(
                            data=scaled_targets,
                            dims=d_names,
                        ),
                    }
                )

        if not dict_to_save:
            return

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
