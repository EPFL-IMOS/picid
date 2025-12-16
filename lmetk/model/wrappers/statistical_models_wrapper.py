from einops import rearrange
import torch
import logging
from typing import Dict, Any, List, Optional

from lmetk.baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from lmetk.model.wrappers.base import AbstractFeedForwardTrainingWrapper
from lmetk.model.methods.statistical_models import (
    LinearBaseline,
    PolynomialBaseline,
    ExponentialBaseline,
)

logger = logging.getLogger(__name__)


class StatisticalBaselineWrapper(AbstractFeedForwardTrainingWrapper):
    """
    A unified wrapper for training statistical baselines within the LMETK framework.

    This wrapper orchestrates the initialization of Linear, Polynomial, and
    Exponential baselines and handles the necessary data reshaping to support:
    1. Univariate Regression
    2. Multivariate Regression
    3. Classification
    """

    def __init__(
        self, task_type: str, seq_len: int, model_type: str = "linear", **kwargs
    ):
        """
        Initializes the wrapper and the underlying statistical backbone.

        Args:
            task_type (str): The specific task name (e.g., 'rul_regression', 'fault_classification').
            seq_len (int): The length of the input time series.
            model_type (str, optional): The type of baseline to use ('linear', 'polynomial', 'exponential').
            **kwargs: Additional configuration (must include 'input_channels').
                - For Classification: Must include 'num_classes'.
                - For Regression: May include 'num_targets' (default=1).
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
        # We explicitly separate 'num_targets' (regression) from 'num_classes' (classification)
        model_task_type = ""
        target_dim = 1

        if task_type in self.regression_tasks:
            model_task_type = "regression"

            # Priority: 'num_targets' -> 'output_dim' -> Default (1)
            if "num_targets" in kwargs:
                target_dim = kwargs.pop("num_targets")
            elif "output_dim" in kwargs:
                target_dim = kwargs.pop("output_dim")
            else:
                target_dim = 1

        elif task_type in self.classification_tasks:
            model_task_type = "classification"
            if "num_classes" not in kwargs:
                raise KeyError(
                    "Configuration Error: 'num_classes' must be provided for classification tasks."
                )
            target_dim = kwargs.pop("num_classes")

        # --- 2. Extract Configuration ---
        if "input_channels" not in kwargs:
            raise KeyError("Configuration Error: 'input_channels' is required.")

        config = {
            "input_channels": kwargs.pop("input_channels"),
            "seq_len": seq_len,
            "poly_degree": kwargs.pop("poly_degree", 2),
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
        Performs the forward pass and reshapes outputs to match framework expectations.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing:
                - 'features': Tensor of shape (Batch, Seq_Len, Channels).
                - '{task_type}': Target tensor.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with 'predictions' and 'targets'.
                - Regression Predictions: (Batch, 1, num_targets)
                - Classification Predictions: (Batch, 1, num_classes)
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
            targets = batch_y

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

            if predictions.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"[Classification] Batch size mismatch: Predictions {predictions.shape[0]} "
                    f"vs Targets {targets.shape[0]}."
                )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        # Allign output format
        targets = targets.unsqueeze(1)

        assert (
            predictions.shape == targets.shape
        ), f"Final shape mismatch: Predictions {predictions.shape} vs Targets {targets.shape}."
        model_out = {
            "predictions": predictions,
            "targets": targets,
        }

        return model_out
