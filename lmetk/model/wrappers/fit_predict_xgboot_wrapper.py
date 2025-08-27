from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from lmetk.model.wrappers.base import AbstractFitPredictWrapper


class FitPredictXGBoostWrapper(AbstractFitPredictWrapper):
    r"""Wrapper for the XGboost model."""

    def __init__(self, **kwargs):
        backbone = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, verbose=3)
        )
        super().__init__(backbone=backbone, **kwargs)
