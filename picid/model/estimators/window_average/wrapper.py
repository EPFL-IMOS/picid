"""Window-average wrapper."""

from picid.model.definitions import FORECASTING_TASKS
from picid.model.adapters.base import AbstractFeedForwardWrapper
from .model import WindowAverageBaseline


class WindowAverageWrapper(AbstractFeedForwardWrapper):
    """
    Wrap the window-average forecasting baseline in the model interface.

    Parameters
    ----------
    task_type : str
        Forecasting task identifier.
    **kwargs : Any
        Additional wrapper arguments forwarded to the baseline.
    """

    def __init__(self, task_type: str, **kwargs):
        """
        Initialize the window-average wrapper for forecasting tasks.

        Parameters
        ----------
        task_type : str
            Forecasting task identifier.
        **kwargs : Any
            Additional wrapper arguments forwarded to the baseline.
        """
        self.regression_tasks = FORECASTING_TASKS
        self.classification_tasks = None
        self.supported_types = self.regression_tasks

        if task_type not in self.supported_types:
            raise ValueError(
                f"Task {task_type} not supported for WindowAverageWrapper."
            )

        self.task_type = task_type

        backbone = WindowAverageBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """
        Run the window-average baseline on one batch.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing ``predictions`` and ``targets``.
        """
        batch_x = batch["target"]["target_seq_x"]
        batch_y = batch["target"]["target_seq_y"]

        outputs = self.backbone(batch_x)
        outputs = outputs[:, -self.kwargs.pred_len :, :]
        batch_y = batch_y[:, -self.kwargs.pred_len :, :]

        return {"predictions": outputs, "targets": batch_y}


__all__ = ["WindowAverageWrapper"]
