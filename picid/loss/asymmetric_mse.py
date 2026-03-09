"""
Asymmetric MSE Loss implementation with Debugging.
"""

from typing import Dict
import torch
import torch.nn as nn

from .base import AbstractLoss


class AsymmetricMSELoss(AbstractLoss):
    """
    Asymmetric Mean Squared Error Loss for RUL prediction.

    Features:
    - Penalizes overestimation (late prediction) more heavily than underestimation.
    - Includes DEBUG prints to verify if asymmetry is actually being triggered.
    """

    def __init__(
        self,
        overestimate_penalty: float = 10.0,
        reduction: str = "mean",
        debug: bool = False,
    ) -> None:
        """
        Initialize AsymmetricMSELoss.

        Parameters
        ----------
        overestimate_penalty :
            Alpha penalty for overestimation (y_pred > y_true).
            Default 10.0 means 10x penalty for dangerous errors.
        reduction :
            'mean', 'sum', or 'none'.
        debug :
            If True, prints the percentage of samples triggering the penalty per batch.
        """
        super().__init__()
        self.alpha = float(overestimate_penalty)
        self.reduction = reduction
        self.debug = debug

        if self.alpha < 1.0:
            raise ValueError(f"overestimate_penalty must be >= 1.0, got {self.alpha}")

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Asymmetric MSE loss.
        """
        # 1. Extract Predictions
        predictions = model_out["predictions"]

        # 2. Extract Targets (Check model_out first, fallback to batch)
        if "targets" in model_out:
            targets = model_out["targets"]
        else:
            targets = batch["targets"]

        # 3. Shape Harmonization (Crucial for Regression)
        # Ensure predictions and targets have the exact same shape (Batch, Seq)

        # Squeeze last dim if it is 1 (e.g. [32, 100, 1] -> [32, 100])
        if predictions.dim() == 3 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        if targets.dim() == 3 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)

        # Ensure float types
        if targets.dtype != predictions.dtype:
            targets = targets.to(dtype=predictions.dtype)

        # 4. Calculate Residuals
        # diff > 0 means Prediction > Target (Dangerous Overestimation)
        diff = predictions - targets

        # 5. Standard MSE (base)
        squared_error = diff**2

        # --- DEBUG BLOCK START ---
        if self.debug:
            with torch.no_grad():
                # Count how many individual points are being penalized
                # diff > 0 is the condition for the penalty
                n_penalized = (diff > 0).sum().item()
                n_total = diff.numel()
                ratio = n_penalized / n_total

                # Only print if this specific batch looks weird or just periodically
                # For now, printing every step to diagnose your "identical loss" issue.
                if ratio == 0.0:
                    print(
                        f"[AsymLoss] Rate: 0.00% (Pure MSE behavior - Model is underestimating safely)"
                    )
                else:
                    print(f"[AsymLoss] Rate: {ratio:.2%} (Penalty Active)")
        # --- DEBUG BLOCK END ---

        # 6. Apply Asymmetric Weights
        # Create a weight tensor: alpha if diff > 0, else 1.0
        weights = torch.where(
            diff > 0,
            torch.tensor(self.alpha, device=diff.device, dtype=diff.dtype),
            torch.tensor(1.0, device=diff.device, dtype=diff.dtype),
        )

        # 7. Weighted Loss Calculation
        weighted_squared_error = weights * squared_error

        if self.reduction == "mean":
            loss = torch.mean(weighted_squared_error)
        elif self.reduction == "sum":
            loss = torch.sum(weighted_squared_error)
        else:
            loss = weighted_squared_error

        result = model_out.copy()
        result["loss"] = loss
        return result
