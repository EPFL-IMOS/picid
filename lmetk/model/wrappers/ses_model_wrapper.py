"""Wrapper for the LinearRegression model."""

from .base import AbstractFeedForwardWrapper

from lmetk.model.methods.ses_model import SESModelBaseline


class SESModelWrapper(AbstractFeedForwardWrapper):
    r"""Wrapper for the SESModel baseline.

    This wrapper defines the forward pass of the SESModel baseline.
    """

    def __init__(self, **kwargs):
        backbone = SESModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        r"""Forward pass for the SES wrapper.

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

        outputs = self.backbone(batch_x)

        # Process outputs following your pattern

        outputs = outputs[:, -self.kwargs.pred_len :, :]
        batch_y = batch_y[:, -self.kwargs.pred_len :, :]

        model_out = {"predictions": outputs, "targets": batch_y}

        return model_out
