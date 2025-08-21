import numpy as np
import torch
from einops import rearrange
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from lmetk.model.wrappers.base import FitPredictWrapper


class FitPredictXGBoostWrapper(FitPredictWrapper):
    r"""Wrapper for the XGboost model."""

    def __init__(self, **kwargs):
        backbone = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=42, verbose=3)
        )
        super().__init__(backbone=backbone, **kwargs)
