import torch
import numpy as np
from typing import List

# Import your new model class
from lmetk.model.methods.cnn_1d import EncoderModel
from lmetk.model.wrappers.base import AbstractFeedForwardTrainingWrapper
import logging

logger = logging.getLogger(__name__)


# (Make sure the _calculate_receptive_field function is defined here or imported)
def _calculate_receptive_field(
    kernels: List[int], strides: List[int], dilations: List[int]
) -> int:
    receptive_field = 1
    total_stride = 1
    if not (len(kernels) == len(strides) == len(dilations)):
        raise ValueError(
            "kernels, strides, and dilations lists must be the same length."
        )
    for k, s, d in zip(kernels, strides, dilations):
        receptive_field += (k - 1) * d * total_stride
        total_stride *= s
    return receptive_field


class CNN1D_Wrapper(AbstractFeedForwardTrainingWrapper):
    """
    Wrapper for the EncoderModel, handling both regression and classification tasks.
    """

    def __init__(self, task_type: str, seq_len: int, **kwargs):

        # Define supported tasks based on your new categories
        self.regression_tasks = ["rul", "ahrul", "soc"]
        self.classification_tasks = ["classification"]
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for CNN1D_Wrapper.")

        self.task_type = task_type
        self.seq_len = seq_len

        # --- Model Configuration ---

        # 1. Determine the model's task_type and num_classes
        if task_type in self.regression_tasks:
            model_task_type = "regression"
            # num_classes is 1 for regression; pop it if user provided it
            num_classes = 1
            kwargs.pop("num_classes", None)
        else:  # task_type in self.classification_tasks
            model_task_type = "classification"
            # For classification, we *require* num_classes from kwargs
            if "num_classes" not in kwargs:
                raise KeyError(
                    "Missing 'num_classes' in kwargs for classification task."
                )
            num_classes = kwargs.pop("num_classes")

        # 2. Extract config keys from kwargs
        config_keys = [
            "input_channels",
            "latent_dim",
            "dropout_prob",
            "output_channels",
            "kernels",
            "strides",
            "dilations",
        ]
        config = {}
        for key in config_keys:
            if key not in kwargs:
                raise KeyError(f"Missing required model config key: '{key}' in kwargs.")
            config[key] = kwargs.pop(key)

        config["input_seq_len"] = seq_len

        # 3. Check Receptive Field (unchanged)
        receptive_field = _calculate_receptive_field(
            config["kernels"], config["strides"], config["dilations"]
        )
        if seq_len > receptive_field:
            raise ValueError(
                f"Configuration Error: Input 'seq_len' ({seq_len}) is larger than "
                f"the model's receptive field ({receptive_field})."
            )
        else:
            logger.info(
                f"Model receptive field ({receptive_field}) is sufficient for "
                f"input seq_len ({seq_len})."
            )

        # 4. Instantiate the new backbone
        backbone = EncoderModel(
            config=config,
            task_type=model_task_type,
            num_classes=num_classes,
        )

        # 5. Clean up old kwargs and pass the rest to super()
        kwargs.pop("cls_head", None)  # No longer used
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """
        Perform a forward pass, now handling both regression and classification.
        """
        # (batch_size, seq_len, in_channels) -> (batch_size, in_channels, seq_len)
        batch_x = batch["features"].permute(0, 2, 1)
        batch_y = batch[self.task_type]  # Get the target tensor

        # The backbone *always* returns the single, correct output tensor
        outputs = self.backbone(batch_x)

        if self.task_type in self.regression_tasks:
            # Model output is (batch_size,)
            # Target is (batch_size, 1)
            # .unsqueeze(1) changes shape to (batch_size, 1)
            predictions = outputs.unsqueeze(1)
            targets = batch_y

            # Shape check for regression
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"[Regression] Shape mismatch: predictions {predictions.shape} "
                    f"and targets {targets.shape}."
                )

        else:  # self.task_type in self.classification_tasks
            # Model output is (batch_size, num_classes)
            predictions = outputs
            targets = batch_y

            # Basic shape check for classification
            if predictions.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"[Classification] Batch size mismatch: predictions {predictions.shape[0]} "
                    f"and targets {targets.shape[0]}."
                )
            # Note: We don't check full shape, as targets can be
            # (batch_size,) for CrossEntropy or (batch_size, num_classes) for BCE.

        model_out = {
            "predictions": predictions,
            "targets": targets,
        }

        return model_out
