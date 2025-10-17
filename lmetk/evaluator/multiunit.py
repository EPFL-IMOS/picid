"""
Multiunit evaluator implementation.

This module provides an evaluator class that inherits from AbstractEvaluator
and implements evaluation logic for transformer-based forecasting models.
"""

import logging

from typing import Any, Dict, List, Optional, override
import numpy as np


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
            get_key = lambda unit: int(unit)
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

            unit_predictions = predictions[:, indices]
            unit_targets = targets[:, indices]

            # Update all metrics associated with this unit key
            for metric in self.unit_metrics[key].values():
                metric.update(unit_predictions, unit_targets)

    @override
    def compute(
        self, mode: str = "val", epoch: int = 0, step: int = 0
    ) -> Dict[str, float]:
        """Compute final metrics, save predictions, and return flattened results."""
        # --- Saving logic from DefaultEvaluator (simplified for example) ---
        if self.save_predictions and self.collect_predictions:
            pass
            # print("\n[Evaluator] Saving predictions to a file would happen here.")
            # # Dummy saving logic:
            # all_preds = np.concatenate(self.preds, axis=0)
            # all_targets = np.concatenate(self.targets, axis=0)
            # print(f"  - Total raw predictions collected: {all_preds.shape}")
            # if self.apply_inverse_scaling:
            #     all_scaled_preds = np.concatenate(self.scaled_preds, axis=0)
            #     print(
            #         f"  - Total scaled predictions collected: {all_scaled_preds.shape}"
            #     )

        # --- Metric computation logic from MultiUnitEvaluator ---
        flat_results = {}
        average_collector = defaultdict(list)

        if not self.unit_metrics:
            return {f"{name}_mean": 0.0 for name in self.metric_names}

        for unit_id, metrics in self.unit_metrics.items():
            for name, metric in metrics.items():
                value = metric.compute()
                flat_results[f"{name}_{unit_id}"] = value
                average_collector[name].append(value)

        for name, values in average_collector.items():
            flat_results[f"{name}_mean"] = np.mean(values)

        for name, values in average_collector.items():
            flat_results[f"{name}_std"] = np.std(values)

        return flat_results
