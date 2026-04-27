import logging
import warnings

import torch
from einops import rearrange
from typing import Any, Dict, Optional

from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper
from .model import (
    LinearBaseline,
    PolynomialBaseline,
    ExponentialBaseline,
)

logger = logging.getLogger(__name__)


class StatisticalBaselineWrapper(AbstractFeedForwardTrainingWrapper):
    """
    Wrap statistical baselines within the PICID framework.

    This wrapper orchestrates the initialization of Linear, Polynomial, and
    Exponential baselines and handles the necessary data reshaping to support:
    1. Univariate Regression
    2. Multivariate Regression
    3. Classification

    Parameters
    ----------
    task_type : str
        Project task identifier.
    seq_len : int
        Input sequence length.
    model_type : str, default="linear"
        Statistical backbone to instantiate.
    input_channels : int
        Number of input channels.
    num_targets : int, optional
        Number of regression targets.
    num_classes : int, optional
        Number of classes for classification tasks.
    output_dim : int, optional
        Deprecated regression target dimension alias.
    poly_degree : int, default=2
        Polynomial expansion degree for the polynomial baseline.
    **kwargs : Any
        Additional wrapper arguments retained by the base wrapper.
    """

    def __init__(
        self,
        task_type: str,
        seq_len: int,
        model_type: str = "linear",
        *,
        input_channels: int,
        num_targets: Optional[int] = None,
        num_classes: Optional[int] = None,
        output_dim: Optional[int] = None,
        poly_degree: int = 2,
        **kwargs: Any,
    ):
        """
        Initialize the wrapper and the underlying statistical backbone.

        Parameters
        ----------
        task_type : str
            Project task identifier.
        seq_len : int
            Input sequence length.
        model_type : str
            Statistical backbone to instantiate.
        input_channels : int
            Number of input channels.
        num_targets : int, optional
            Number of regression targets.
        num_classes : int, optional
            Number of classes for classification tasks.
        output_dim : int, optional
            Deprecated regression target dimension alias.
        poly_degree : int, default=2
            Polynomial expansion degree for the polynomial baseline.
        **kwargs : Any
            Additional wrapper arguments retained by the base wrapper.
        """
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task '{task_type}' is not supported by StatisticalBaselineWrapper."
            )

        self.task_type = task_type
        self.seq_len = seq_len
        self.model_type = model_type.lower()

        # --- 1. Determine Output Dimensions ---
        model_task_type = ""
        target_dim: int = 1

        if task_type in self.regression_tasks:
            model_task_type = "regression"
            if num_targets is not None:
                target_dim = num_targets
            elif output_dim is not None:
                warnings.warn(
                    "'output_dim' is deprecated; use 'num_targets' for regression.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                target_dim = output_dim
            else:
                target_dim = 1

        elif task_type in self.classification_tasks:
            model_task_type = "classification"
            if num_classes is None:
                raise KeyError(
                    "Configuration Error: 'num_classes' must be provided for classification tasks."
                )
            target_dim = num_classes

        # --- 2. Build Configuration ---
        config = {
            "input_channels": input_channels,
            "seq_len": seq_len,
            "poly_degree": poly_degree,
        }

        # --- 3. Instantiate Backbone ---
        if self.model_type == "linear":
            backbone = LinearBaseline(
                config, task_type=model_task_type, num_targets=target_dim
            )
        elif self.model_type == "polynomial":
            backbone = PolynomialBaseline(
                config, task_type=model_task_type, num_targets=target_dim
            )
        elif self.model_type == "exponential":
            backbone = ExponentialBaseline(
                config, task_type=model_task_type, num_targets=target_dim
            )
        else:
            raise ValueError(
                f"Unknown model_type: '{self.model_type}'. Supported: linear, polynomial, exponential."
            )

        logger.info(
            f"Initialized Statistical Baseline: {self.model_type.capitalize()} "
            f"| Task: {model_task_type} | Output Dim: {target_dim}"
        )

        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform the forward pass and reshape outputs to match framework expectations.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Batch dictionary containing features and the task target.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with ``predictions`` and ``targets``.

            - Regression predictions have shape ``(batch, 1, num_targets)``.
            - Classification predictions have shape ``(batch, 1, num_classes)``.
        """
        # (batch_size, seq_len, in_channels) -> (batch_size, in_channels, seq_len)
        batch_x = batch["features"].permute(0, 2, 1)
        batch_y = batch[self.task_type]

        # Backbone Output is ALWAYS (Batch, target_dim)
        outputs = self.backbone(batch_x)

        if self.task_type in self.regression_tasks:
            # Standardize regression output to: (Batch, 1, num_targets)
            # This handles both Univariate (num_targets=1) and Multivariate (num_targets>1)
            predictions = rearrange(outputs, "b c -> b 1 c")
            targets = batch_y.unsqueeze(1)

            assert (
                predictions.shape == targets.shape
            ), f"Final shape mismatch: Predictions {predictions.shape} vs Targets {targets.shape}."

            # --- Shape Safety Check ---
            pred_dim = predictions.shape[-1]
            target_dim = targets.shape[-1] if targets.dim() > 1 else 1

            if pred_dim != target_dim:
                # Warning: Broadcasting might still allow this to run, but strictly speaking
                # the dimensions should match for multivariate regression.
                logger.debug(
                    f"[Regression] Dimension mismatch: Model predicts {pred_dim} targets, "
                    f"but ground truth has {target_dim}."
                )

        elif self.task_type in self.classification_tasks:
            # Classification Output: (Batch, num_classes) -> (Batch, 1, num_classes)
            predictions = rearrange(outputs, "b c -> b 1 c")

            # Targets: Usually (Batch,) or (Batch, 1).
            # We unsqueeze to ensure time-dimension consistency.
            targets = batch_y.unsqueeze(1)

            # Check that first two dims are matching
            assert (
                predictions.shape[0] == targets.shape[0]
            ), f"Batch size mismatch: Predictions {predictions.shape[0]} vs Targets {targets.shape[0]}."

            if predictions.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"[Classification] Batch size mismatch: Predictions {predictions.shape[0]} "
                    f"vs Targets {targets.shape[0]}."
                )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        model_out = {
            "predictions": predictions,
            "targets": targets,
        }

        return model_out


__all__ = ["StatisticalBaselineWrapper"]
