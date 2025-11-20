from typing import List, Optional, override
from lmetk.transforms.base.multisource import InverseTransformMixin
from lmetk.evaluator.default import DefaultEvaluator


class ForecastingEvaluator(DefaultEvaluator):

    def __init__(
        self,
        target_dim_position: Optional[int],
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

        # Note: self.task_mode is not defined in init args in your snippet,
        # assuming it comes from kwargs or is set elsewhere.
        # For this implementation, I'll look for it in kwargs or default to None.
        self.task_mode = kwargs.get("task_mode", "univariate")

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

        # 1. Slice by effective prediction length
        if self.effective_pred_len is not None:
            predictions = predictions[:, -self.effective_pred_len :, :]
            targets = targets[:, -self.effective_pred_len :, :]

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )

        # 2. Select specific target dimension if Univariate task
        if self.task_mode == "univariate" and self.target_dim_position is not None:
            f_dim = self.target_dim_position
            # Slicing preserves dim if we don't squeeze, but usually metrics expect (B, T) or (B, T, 1)
            # Your logic: predictions[:, :, f_dim] returns (B, T).
            # To keep metrics happy (often expect 3D), we might want to keep dims,
            # but following your code exactly:
            predictions = predictions[:, :, f_dim]
            targets = targets[:, :, f_dim]

        model_out["predictions"] = predictions
        model_out["targets"] = targets
        super().update(model_out)
