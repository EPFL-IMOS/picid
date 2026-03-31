"""
Cross Entropy Loss implementation.
"""

from typing import Dict
import torch
import torch.nn as nn

from .base import AbstractLoss


class CrossEntropyLoss(AbstractLoss):
    """
    Cross Entropy loss for classification tasks.

    This loss combines LogSoftmax and NLLLoss in one single class.
    It expects raw logits from the model (unnormalized scores).

    Parameters
    ----------
    reduction : str, default="mean"
        Specifies the reduction to apply to the output: ``"none"``, ``"mean"``, or ``"sum"``.
    ignore_index : int, default=-100
        Specifies a target value that is ignored and does not contribute to the gradient.
    label_smoothing : float, default=0.0
        Specifies the amount of smoothing in ``[0.0, 1.0]``.
    """

    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Initialize the cross-entropy loss wrapper.

        Parameters
        ----------
        reduction : str, default="mean"
            Specifies the reduction to apply to the output: ``"none"``, ``"mean"``, or ``"sum"``.
        ignore_index : int, default=-100
            Specifies a target value that is ignored and does not contribute to the gradient.
        label_smoothing : float, default=0.0
            Specifies the amount of smoothing in ``[0.0, 1.0]``.
        """
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            reduction=reduction,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the cross-entropy loss for classification outputs.

        Parameters
        ----------
        model_out : Dict[str, torch.Tensor]
            Dictionary with ``"predictions"`` of shape ``(batch, seq, num_classes)``.
        batch : Dict[str, torch.Tensor]
            Dictionary with ``"targets"`` of shape ``(batch, seq)`` or ``(batch, seq, 1)``.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with the original model output plus a ``"loss"`` key.
        """
        # Predictions: (Batch, Seq, Classes)
        predictions = model_out["predictions"]
        # Targets: (Batch, Seq) or (Batch, Seq, 1)
        targets = model_out["targets"]

        # --- Reshape for PyTorch CrossEntropyLoss ---
        # PyTorch expects:
        #   Input: (Batch, Classes, Seq)
        #   Target: (Batch, Seq) containing class indices (Long)

        # 1. Permute Predictions: (Batch, Seq, Classes) -> (Batch, Classes, Seq)
        if predictions.dim() == 3:
            predictions = predictions.permute(0, 2, 1)

        # 2. Handle Targets
        # Case A: Targets are 3D (Batch, Seq, 1) -> Squeeze to (Batch, Seq)
        if targets.dim() == 3 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        # Case B: Targets are already 2D (Batch, Seq) -> Do nothing

        # 3. Ensure Targets are Long (Class Indices)
        if targets.dtype not in [torch.long, torch.int64]:
            targets = targets.long()

        loss = self.loss_fn(predictions, targets)

        result = model_out.copy()
        result["loss"] = loss
        return result
