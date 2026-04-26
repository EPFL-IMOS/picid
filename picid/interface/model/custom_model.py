from typing import Callable

import torch
from einops import rearrange
from torch import nn

from picid.model.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper


def default_pre_process(
    batch: dict[str, torch.Tensor], to_extract: str | list[str], **kwargs
) -> list[torch.Tensor]:
    """
    Extracts specified tensors from a batch dictionary.

    This utility function retrieves data from a batch based on the provided
    keys. It normalizes the input so that whether a single string or a list
    of strings is requested, it consistently returns a list of the
    corresponding tensors.

    Args:
        batch (dict[str, torch.Tensor]): A dictionary containing the batch data,
            where keys are strings (e.g., 'features', 'labels') and values
            are PyTorch tensors.
        to_extract (str | list[str]): A single key or a list of keys indicating
            which tensors to pull from the batch dictionary.
        **kwargs: Additional keyword arguments (currently unused in this
            default implementation).

    Returns:
        list[torch.Tensor]: A list of extracted tensors ordered according to
            the keys provided in `to_extract`.

    Raises
    ------
    KeyError
        If any key in ``to_extract`` is not present in ``batch``.
    """

    if isinstance(to_extract, str):
        to_extract = [to_extract]

    return [batch[t] for t in to_extract]


class CustomModelTrainer(AbstractFeedForwardTrainingWrapper):
    """
    A wrapper class for feed-forward neural network models during training.

    This class handles the pre-processing of input batches, routes the data
    through the underlying model (referred to as the backbone), and ensures
    that the shapes of the predictions and targets are properly aligned
    based on whether the task is a regression or classification problem.

    Attributes:
        task_type (str): The specific type of machine learning task (e.g., 'regression', 'classification').
        _pre_process_function (Callable): The function used to extract features and targets from the raw batch.
    """

    def __init__(
        self,
        task_type: str,
        model: nn.Module,
        pre_process_function: Callable | None = default_pre_process,
        keys_to_extract: list[str] | None = None,
        **kwargs,
    ):
        """
        Initializes the CustomModelTrainer.

        Args:
            task_type (str): The type of task being performed. Expected to be a value present
                in either REGRESSION_TASKS or CLASSIFICATION_TASKS.
            model (nn.Module): The core PyTorch neural network model to be wrapped and trained.
            pre_process_function (Callable | None, optional): A function to process the incoming
                batch dictionary into inputs (x) and targets (y). Defaults to `default_pre_process`.
            keys_to_extract (list[str] | None, optional): The keys to extract from the batch dict.
                Defaults to ``['features', task_type]`` when ``None``.
            **kwargs: Additional keyword arguments to be passed to the parent class.
        """

        # add checks on task type
        self.task_type = task_type
        self._pre_process_function = pre_process_function
        if keys_to_extract is None:
            keys_to_extract = ["features", self.task_type]

        self._keys_to_extract = keys_to_extract

        super().__init__(model, **kwargs)

    def forward(
        self, batch: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Performs the forward pass of the model.

        Extracts the input features and target values from the batch, runs the inputs
        through the model backbone, and reshapes the outputs and targets to ensure
        they have a matching 3D shape `(batch_size, 1, channels)` for loss calculation.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing the batch data. It must
                contain the keys listed in `self._keys_to_extract` (by default
                ``'features'`` and the value of `self.task_type``).
            **kwargs: Additional keyword arguments passed to the pre-process function
                and the model backbone.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing two keys:
                - "predictions": The reshaped output from the model.
                - "targets": The reshaped ground truth targets.

        Raises:
            ValueError: If `self.task_type` is not found in `REGRESSION_TASKS` or `CLASSIFICATION_TASKS`.
        """

        x, y = self._pre_process_function(
            batch, to_extract=self._keys_to_extract, **kwargs
        )

        outputs = self.backbone(x, batch=batch, **kwargs)

        if self.task_type in REGRESSION_TASKS:
            if len(outputs.shape) == 2:
                outputs = rearrange(outputs, "b c -> b 1 c")
            elif len(outputs.shape) == 1:
                outputs = rearrange(outputs, "b -> b 1 1")
            y = y.unsqueeze(1)

        elif self.task_type in CLASSIFICATION_TASKS:
            if len(outputs.shape) == 2:
                outputs = rearrange(outputs, "b c -> b 1 c")
            y = y.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        model_out = {
            "predictions": outputs,
            "targets": y,
        }

        return model_out
