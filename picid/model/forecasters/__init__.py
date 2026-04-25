"""Canonical forecaster namespace backed by legacy baseline implementations."""

from picid.model.forecasters.crossformer_model import Crossformer_Forecaster
from picid.model.forecasters.forecaster import Forecaster, TransformerForecaster
from picid.model.forecasters.lstm_model import LSTM_Forecaster
from picid.model.forecasters.patchtst_model import PatchTST_Forecaster
from picid.model.forecasters.spacetimeformer_model import Spacetimeformer_Forecaster
from picid.model.forecasters.tide_model import TiDE_Forecaster
from picid.model.forecasters.timeseries_transformer_model import (
    Timeseries_Transformer_Forecaster,
)

__all__ = [
    "Forecaster",
    "TransformerForecaster",
    "LSTM_Forecaster",
    "PatchTST_Forecaster",
    "Crossformer_Forecaster",
    "TiDE_Forecaster",
    "Timeseries_Transformer_Forecaster",
    "Spacetimeformer_Forecaster",
]
