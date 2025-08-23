"""Wrapper for the TimePFN model."""

import torch
from tabpfn import TabPFNRegressor


from .base import AbstractWrapper


class BatchContextTabPFNWrapper(AbstractWrapper):
    r"""Wrapper for the TimePFN model.

    This wrapper defines the forward pass of the TimePFN model.
    """

    def __init__(self, **kwargs):
        backbone = TabPFNRegressor(
            ignore_pretraining_limits=kwargs["ignore_pretraining_limits"],
            random_state=42,
            device="cuda" if kwargs["device"] == "gpu" else "cpu",
        )
        super().__init__(backbone=backbone, **kwargs)
        self.model_fit_flag = False
        self.lag = 0

        # self.train_dataloader = kwargs['train_dataloader']
        # self.val_dataloader = kwargs['val_dataloader']

        self.aux_context = None
        self.y_aux = None

    def fit(self, x, y):
        self.backbone.fit(x.cpu(), y.cpu())

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

        context_fit = batch["context"]["x"].squeeze(0)
        target_fit = batch["target"]["x"].squeeze(0)

        context = batch["context"]["y"].squeeze(0)
        target = batch["target"]["y"].squeeze(0)

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
