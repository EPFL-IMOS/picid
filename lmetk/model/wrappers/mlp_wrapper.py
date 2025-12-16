from einops import rearrange
import torch
import logging
from typing import Dict, Any

from lmetk.baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from lmetk.model.wrappers.base import AbstractFeedForwardTrainingWrapper
from lmetk.model.methods.mlp import MLP

logger = logging.getLogger(__name__)


class MLPWrapper(AbstractFeedForwardTrainingWrapper):
    """
    Wrapper specifically for the MLP Baseline model.
    """

    def __init__(self, task_type: str, seq_len: int, **kwargs):
        """
        Args:
            task_type (str): Task name.
            seq_len (int): Input sequence length.
            **kwargs:
                - 'input_channels' (required)
                - 'num_targets' (required)
                - 'hidden_dim' (optional, default 64)
                - 'num_layers' (optional, default 2)
        """
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task '{task_type}' is not supported by MLPWrapper.")

        self.task_type = task_type

        # --- 1. Determine Dimensions ---
        target_dim = 1
        model_task_type = ""

        if task_type in self.regression_tasks:
            model_task_type = "regression"
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
                    "Configuration Error: 'num_classes' must be provided for classification."
                )
            target_dim = kwargs.pop("num_classes")

        # --- 2. Extract Config ---
        if "input_channels" not in kwargs:
            raise KeyError("Configuration Error: 'input_channels' is required.")

        config = {
            "input_channels": kwargs.pop("input_channels"),
            "seq_len": seq_len,
            "hidden_dim": kwargs.pop("hidden_dim", 64),
            "num_layers": kwargs.pop("num_layers", 2),  # <--- Extract num_layers
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
