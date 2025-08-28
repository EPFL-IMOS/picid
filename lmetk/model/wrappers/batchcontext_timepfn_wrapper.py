"""Wrapper for the TimePFN model."""

import torch
from lightning_fabric.utilities.data import AttributeDict

from lmetk.model.time_pfn.TimePFN import TimePFN

from .base import AbstractFeedForwardWrapper


class BatchContextTimePFNWrapper(AbstractFeedForwardWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, **kwargs):
        backbone = TimePFN(AttributeDict(kwargs))
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
        batch_x_mark = batch["batch_x_mark"]
        batch_y_mark = batch["batch_y_mark"]

        channel_size = batch_x.size(2)
        max_channel_size = 161
        number_of_iter = channel_size // max_channel_size + 1

        pred = None
        true = None

        for j in range(number_of_iter):
            batch_x_subset = batch_x[
                :, :, j * max_channel_size : (j + 1) * max_channel_size
            ]
            batch_y_subset = batch_y[
                :, :, j * max_channel_size : (j + 1) * max_channel_size
            ]

            outputs = self.backbone(batch_x_subset, batch_x_mark)
            outputs = outputs[:, -self.kwargs.pred_len :, :]
            batch_y_subset = batch_y_subset[:, -self.kwargs.pred_len :, :]

            if pred is None:
                pred = outputs
                true = batch_y_subset
            else:
                pred = torch.concat((pred, outputs), dim=-1)
                true = torch.concat((true, batch_y_subset), axis=-1)

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {"predictions": outputs, "targets": batch_y_subset}

        return model_out
