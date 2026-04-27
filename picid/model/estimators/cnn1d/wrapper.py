import logging
from typing import Any, List, Optional

from einops import rearrange

from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper

from .model import EncoderModel

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
    Wrap the CNN encoder for regression and classification tasks.

    Parameters
    ----------
    task_type : str
        Project task identifier.
    seq_len : int
        Input sequence length.
    input_channels : int
        Number of input channels.
    latent_dim : int
        Latent representation size.
    dropout_prob : float
        Dropout probability used inside the encoder blocks.
    output_channels : int
        Number of channels in the final encoder block.
    kernels : list[int]
        Convolution kernel sizes for each encoder block.
    strides : list[int]
        Convolution strides for each encoder block.
    dilations : list[int]
        Convolution dilations for each encoder block.
    num_classes : int, optional
        Number of classes for classification tasks.
    cls_head : Any, optional
        Legacy classification head argument kept for compatibility.
    **kwargs : Any
        Additional wrapper arguments retained by the base wrapper.
    """

    def __init__(
        self,
        task_type: str,
        seq_len: int,
        *,
        input_channels: int,
        latent_dim: int,
        dropout_prob: float,
        output_channels: int,
        kernels: List[int],
        strides: List[int],
        dilations: List[int],
        num_classes: Optional[int] = None,
        cls_head: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.regression_tasks = REGRESSION_TASKS
        self.classification_tasks = CLASSIFICATION_TASKS
        self.supported_types = self.regression_tasks + self.classification_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for CNN1D_Wrapper.")

        self.task_type = task_type
        self.seq_len = seq_len

        # --- 1. Determine task_type and num_classes ---
        if task_type in self.regression_tasks:
            model_task_type = "regression"
            num_classes_val = 1
        elif task_type in self.classification_tasks:
            model_task_type = "classification"
            if num_classes is None:
                raise KeyError(
                    "Missing 'num_classes' for classification task. "
                    "Pass num_classes=<int>."
                )
            num_classes_val = num_classes
        else:
            raise ValueError(f"Task {task_type} not supported for CNN1D_Wrapper.")

        # --- 2. Build config from explicit args ---
        config = {
            "input_channels": input_channels,
            "latent_dim": latent_dim,
            "dropout_prob": dropout_prob,
            "output_channels": output_channels,
            "kernels": kernels,
            "strides": strides,
            "dilations": dilations,
            "input_seq_len": seq_len,
        }

        # --- 3. Check receptive field ---
        receptive_field = _calculate_receptive_field(
            config["kernels"], config["strides"], config["dilations"]
        )
        if seq_len > receptive_field:
            raise ValueError(
                f"Configuration Error: Input 'seq_len' ({seq_len}) is larger than "
                f"the model's receptive field ({receptive_field})."
            )
        logger.info(
            f"Model receptive field ({receptive_field}) is sufficient for "
            f"input seq_len ({seq_len})."
        )

        # --- 4. Instantiate backbone (cls_head ignored for backward compat) ---
        backbone = EncoderModel(
            config=config,
            task_type=model_task_type,
            num_classes=num_classes_val,
        )
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """
        Run the CNN wrapper on one batch.

        Parameters
        ----------
        batch : dict
            Batch dictionary containing features and targets.

        Returns
        -------
        dict
            Dictionary containing predictions and targets.
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
            predictions = rearrange(outputs, "b -> b 1 1")
            targets = batch_y.unsqueeze(1)  # rearrange(batch_y, "b -> b 1 1")

            # Shape check for regression
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"[Regression] Shape mismatch: predictions {predictions.shape} "
                    f"and targets {targets.shape}."
                )

        elif self.task_type in self.classification_tasks:
            # Model output is (batch_size, num_classes)
            predictions = rearrange(outputs, "b c -> b 1 c")
            targets = batch_y.unsqueeze(1)

            # Basic shape check for classification
            if predictions.shape[0] != targets.shape[0]:
                raise ValueError(
                    f"[Classification] Batch size mismatch: predictions {predictions.shape[0]} "
                    f"and targets {targets.shape[0]}."
                )
            # Note: We don't check full shape, as targets can be
            # (batch_size,) for CrossEntropy or (batch_size, num_classes) for BCE.
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        model_out = {
            "predictions": predictions,
            "targets": targets,
        }

        return model_out


__all__ = ["CNN1D_Wrapper", "_calculate_receptive_field"]
