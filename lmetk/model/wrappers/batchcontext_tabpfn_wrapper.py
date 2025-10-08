"""Wrapper for the TimePFN model."""

import torch
from tabpfn import TabPFNRegressor


from .base import AbstractFeedForwardWrapper


class BatchContextTabPFNWrapper(AbstractFeedForwardWrapper):
    """Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, **kwargs):
        backbone = TabPFNRegressor(
            ignore_pretraining_limits=kwargs["ignore_pretraining_limits"],
            random_state=kwargs["random_state"],
            device="cuda" if kwargs["device"] == "gpu" else "cpu",
        )
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0

        self.aux_context = None
        self.y_aux = None
        self.verify_inputs = False

    def fit(self, x, y):
        self.backbone.fit(x.cpu(), y.cpu())

    def forward(self, batch):
        """Forward pass for the Tune wrapper.

        Parameters
        ----------self.backbone.fit(batch_x, batch_y)
        batch : dict
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        if not self.verify_inputs:
            assert (
                batch["context"]["features_seq_x"].shape[0] == 1
            ), "Dimension 0 of features_seq_x must have exactly 1 element"
            assert (
                batch["target"]["target_seq_x"].shape[0] == 1
            ), "Dimension 0 of target_seq_x must have exactly 1 element"
            assert (
                batch["context"]["features_seq_y"].shape[0] == 1
            ), "Dimension 0 of features_seq_y must have exactly 1 element"
            assert (
                batch["target"]["target_seq_y"].shape[0] == 1
            ), "Dimension 0 of target_seq_y must have exactly 1 element"
            self.verify_inputs = True

        context_fit = batch["context"]["features_seq_x"].squeeze(0)
        target_fit = batch["target"]["target_seq_x"].squeeze(0)

        context = batch["context"]["features_seq_y"].squeeze(0)
        target = batch["target"]["target_seq_y"].squeeze(0)

        # Extent the context with auxiliary features to give sence of seasonality
        self.fit(context_fit, target_fit)

        # Extent the context with auxiliary features to give sence of seasonality
        outputs = self.backbone.predict(context.cpu())

        # Update evaluator with current batch (replaces the accumulation logic)
        model_out = {
            "predictions": torch.Tensor(outputs).reshape(target.shape),
            "targets": target,
        }

        return model_out
