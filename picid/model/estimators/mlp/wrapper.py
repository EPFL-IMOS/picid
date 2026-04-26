import logging
import warnings

import torch
from einops import rearrange
from typing import Any, Dict, Optional

from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper

from .model import MLP

logger = logging.getLogger(__name__)


class MLPWrapper(AbstractFeedForwardTrainingWrapper):
    """
    Wrap the MLP baseline in the training-oriented model interface.

    Parameters
    ----------
    task_type : str
        Project task identifier.
    seq_len : int
        Input sequence length.
    input_channels : int
        Number of input channels.
    num_targets : int, optional
        Number of regression targets.
    num_classes : int, optional
        Number of classes for classification tasks.
    output_dim : int, optional
        Deprecated regression target dimension alias.
    hidden_dim : int, default=64
        Hidden feature size used by the MLP.
    num_layers : int, default=2
        Number of linear blocks in the MLP.
    **kwargs : Any
        Additional wrapper arguments retained by the base wrapper.
    """

    def __init__(
        self,
        task_type: str,
        seq_len: int,
        *,
        input_channels: int,
        num_targets: Optional[int] = None,
        num_classes: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        **kwargs: Any,
    ):
        """
        Create the MLP wrapper and select the correct target dimension.

        Parameters
        ----------
        task_type : str
            Project task identifier.
        seq_len : int
            Input sequence length.
        input_channels : int
            Number of input channels.
        num_targets : int, optional
            Number of regression targets.
        num_classes : int, optional
            Number of classes for classification tasks.
        output_dim : int, optional
            Deprecated regression target dimension alias.
        hidden_dim : int, default=64
            Hidden feature size used by the MLP.
        num_layers : int, default=2
            Number of linear blocks in the MLP.
        **kwargs : Any
            Additional wrapper arguments retained by the base wrapper.
        """
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task '{task_type}' is not supported by MLPWrapper.")

        self.task_type = task_type

        # --- 1. Determine Dimensions ---
        target_dim: int = 1
        model_task_type = ""

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
                    "Configuration Error: 'num_classes' must be provided for classification."
                )
            target_dim = num_classes

        # --- 2. Build Config ---
        config = {
            "input_channels": input_channels,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }

        # --- 3. Instantiate MLP Backbone ---
        backbone = MLP(config, task_type=model_task_type, num_targets=target_dim)

        logger.info(
            f"Initialized MLP Baseline | Layers: {config['num_layers']} | "
            f"Hidden: {config['hidden_dim']} | Task: {model_task_type}"
        )

        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # (batch_size, seq_len, in_channels) -> (batch_size, in_channels, seq_len)
        batch_x = batch["features"].permute(0, 2, 1)
        batch_y = batch[self.task_type]

        outputs = self.backbone(batch_x)

        if self.task_type in self.regression_tasks:
            predictions = rearrange(outputs, "b c -> b 1 c")
            targets = batch_y.unsqueeze(1)

            if predictions.shape[-1] != targets.shape[-1]:
                logger.debug(
                    f"[MLP] Dimension mismatch: Preds {predictions.shape} vs Targets {targets.shape}"
                )

        elif self.task_type in self.classification_tasks:
            predictions = rearrange(outputs, "b c -> b 1 c")
            targets = batch_y.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        model_out = {
            "predictions": predictions,
            "targets": targets,
        }

        return model_out


__all__ = ["MLPWrapper"]
