import logging
from typing import List, Optional, Any, override
import numpy as np
from lmetk.evaluator.default import DefaultEvaluator

logger = logging.getLogger(__name__)


class ForecastingEvaluator(DefaultEvaluator):
    def __init__(
        self,
        target_dim_position: Optional[int],
        metric_names: List[str],
        model_seq_len: int,
        model_label_len: int,
        effective_pred_len: Optional[int] = None,
        model_pred_len: Optional[int] = None,
        inverse_transform: Optional[Any] = None,
        apply_inverse_scaling: bool = False,
        save_predictions: bool = False,
        paths: Optional[Any] = None,
        # task_mode: str = "univariate",
        **kwargs,
    ):
        # 1. Validation Logic
        if model_pred_len is None:
            raise ValueError("Model's pred_len (model_pred_len) must be set.")

        if effective_pred_len is not None and model_pred_len < effective_pred_len:
            raise ValueError(
                f"effective_pred_len ({effective_pred_len}) cannot be greater than "
                f"model_pred_len ({model_pred_len})."
            )

        # 2. Aggressive task_mode resolution
        # We check both the explicit arg and the kwargs to prevent booleans from leaking in
        # legacy_univariate = kwargs.pop("univariate", None)
        self.task_mode = kwargs.get("task_mode", None)
        assert self.task_mode in [
            "univariate",
            "multivariate",
            None,
        ], f"task_mode must be 'univariate', 'multivariate', or None, got {self.task_mode}"

        # 3. Attribute Assignment
        self.model_label_len = model_label_len
        self.model_pred_len = model_pred_len
        self.model_seq_len = model_seq_len
        self.effective_pred_len = effective_pred_len
        self.target_dim_position = target_dim_position

        super().__init__(
            metric_names=metric_names,
            inverse_transform=inverse_transform,
            apply_inverse_scaling=apply_inverse_scaling,
            save_predictions=save_predictions,
            paths=paths,
            task_type="forecasting",
            **kwargs,
        )

    @override
    def update(self, model_out: dict):
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        if self.effective_pred_len is not None:
            predictions = predictions[:, -self.effective_pred_len :, :]
            targets = targets[:, -self.effective_pred_len :, :]

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        if self.task_mode == "univariate" and self.target_dim_position is not None:
            f_dim = self.target_dim_position
            predictions = predictions[:, :, f_dim]
            targets = targets[:, :, f_dim]

        model_out["predictions"] = predictions
        model_out["targets"] = targets
        super().update(model_out)
