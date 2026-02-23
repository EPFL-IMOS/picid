"""Wrapper for the TimePFN model."""

import torch
from picid.model.wrappers.base import AbstractFeedForwardTrainingWrapper
from picid.model.autoencoder.residual_1d import SymmetricalAutoencoder


class AutoencoderWrapper(AbstractFeedForwardTrainingWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, **kwargs):
        backbone = SymmetricalAutoencoder(kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        r"""Forward pass for the Tune wrapper.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        x = batch["features_seq_x"]

        # Extent the context with auxiliary features to give sence of seasonality
        reconstructed_output = self.backbone(x)
        assert x.shape == reconstructed_output.shape

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            "predictions": torch.Tensor(reconstructed_output),
            "targets": x,
        }

        return model_out
