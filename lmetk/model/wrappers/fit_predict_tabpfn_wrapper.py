"""Wrapper for the TimePFN model."""

import torch
from lightning_fabric.utilities.data import AttributeDict
from tabpfn import TabPFNRegressor

from lmetk.model.time_pfn.TimePFN import TimePFN

from .base import FitPredictWrapper


class FitPredictTabPFNWrapper(FitPredictWrapper):
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

    def get_features(self, batch):
        context = batch["context"]["x"]
        return context

    def get_targets(self, batch):
        target = batch["target"]["y"]
        return target
