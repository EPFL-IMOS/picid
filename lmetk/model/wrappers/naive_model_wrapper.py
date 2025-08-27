"""Wrapper for the NaiveModel model."""

from .base import AbstractFeedForwardWrapper
from lmetk.model.methods.naive_model import NaiveModelBaseline


class NaiveModelWrapper(AbstractFeedForwardWrapper):
    r"""Wrapper for the NaiveModel model.

    This wrapper defines the forward pass of the NaiveModel model.
    """

    def __init__(self, **kwargs):
        backbone = NaiveModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        batch_x = batch["batch_x"]
        batch_y = batch["batch_y"]

        # batch_x_mark = batch['batch_x_mark']
        # batch_y_mark = batch['batch_y_mark']

        outputs = self.backbone(batch_x)

        outputs = outputs[:, -self.kwargs.pred_len :, :]
        batch_y = batch_y[:, -self.kwargs.pred_len :, :]

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {"predictions": outputs, "targets": batch_y}

        return model_out
