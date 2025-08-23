"""Wrapper for the TimePFN model."""

from tabpfn import TabPFNRegressor


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
