"""Wrapper for the linear-regression forecasting baseline."""

from .base import AbstractFeedForwardWrapper

from picid.model.methods.linear_regression_model import LinearRegressionModelBaseline
from ...baselines.definitions import FORECASTING_TASKS


class LinearRegressionModelWrapper(AbstractFeedForwardWrapper):
    """
    Wrap the linear-regression baseline in the model interface.

    Parameters
    ----------
    task_type : str
        Forecasting task identifier.
    **kwargs : Any
        Additional wrapper arguments forwarded to the baseline.
    """

    def __init__(self, task_type: str, **kwargs):
        # Define supported tasks based on your new categories
        self.regression_tasks = FORECASTING_TASKS
        self.classification_tasks = None
        self.supported_types = self.regression_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for LinearRegressionModelWrapper."
            )

        self.task_type = task_type
        backbone = LinearRegressionModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """
        Run the linear-regression baseline on one batch.

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

        outputs = self.backbone(batch_x)

        # Process outputs following your pattern
        outputs = outputs[:, -self.kwargs.pred_len :, :]
        batch_y = batch_y[:, -self.kwargs.pred_len :, :]

        model_out = {"predictions": outputs, "targets": batch_y}

        return model_out
