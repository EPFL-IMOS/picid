"""Wrapper for the NaiveModel model."""

from .base import AbstractFeedForwardWrapper
from picid.model.methods.naive_model import NaiveModelBaseline
from ...baselines.definitions import FORECASTING_TASKS


class NaiveModelWrapper(AbstractFeedForwardWrapper):
    """Wrapper for the NaiveModel model.

    This wrapper defines the forward pass of the NaiveModel model.
    """

    def __init__(self, task_type: str, **kwargs):
        # Define supported tasks based on your new categories
        self.regression_tasks = FORECASTING_TASKS
        self.classification_tasks = None
        self.supported_types = self.regression_tasks

        if task_type not in self.supported_types:
            raise ValueError(f"Task {task_type} not supported for NaiveModelWrapper.")

        self.task_type = task_type

        backbone = NaiveModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """Forward pass for the Tune wrapper.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        batch_x = batch["target"]["target_seq_x"]
        batch_y = batch["target"]["target_seq_y"]

        # batch_x_mark = batch['batch_x_mark']
        # batch_y_mark = batch['batch_y_mark']

        outputs = self.backbone(batch_x)

        outputs = outputs[:, -self.kwargs.pred_len :, :]
        batch_y = batch_y[:, -self.kwargs.pred_len :, :]

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {"predictions": outputs, "targets": batch_y}

        return model_out
