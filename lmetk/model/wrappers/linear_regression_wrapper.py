"""Wrapper for the LinearRegression model."""

from .base import AbstractFeedForwardWrapper

from lmetk.model.methods.linear_regression_model import LinearRegressionModelBaseline


class LinearRegressionModelWrapper(AbstractFeedForwardWrapper):
    """Wrapper for the LinearRegressionModel baseline.

    This wrapper defines the forward pass of the LinearRegressionModel baseline.
    """

    def __init__(self, **kwargs):
        backbone = LinearRegressionModelBaseline(**kwargs)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch):
        """Forward pass for the LinearRegression wrapper.

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
