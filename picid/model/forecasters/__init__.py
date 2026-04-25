"""Canonical forecaster namespace.

This package exposes lazy top-level imports so submodules can resolve without
eagerly importing the entire forecaster tree.
"""

from importlib import import_module
from typing import TYPE_CHECKING

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

_SYMBOL_TO_MODULE = {
    "Forecaster": "picid.model.forecasters.forecaster",
    "TransformerForecaster": "picid.model.forecasters.forecaster",
    "LSTM_Forecaster": "picid.model.forecasters.lstm_model",
    "PatchTST_Forecaster": "picid.model.forecasters.patchtst_model",
    "Crossformer_Forecaster": "picid.model.forecasters.crossformer_model",
    "TiDE_Forecaster": "picid.model.forecasters.tide_model",
    "Timeseries_Transformer_Forecaster": "picid.model.forecasters.timeseries_transformer_model",
    "Spacetimeformer_Forecaster": "picid.model.forecasters.spacetimeformer_model",
}


def __getattr__(name):
    if name not in _SYMBOL_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_SYMBOL_TO_MODULE[name])
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.crossformer_model import Crossformer_Forecaster
    from picid.model.forecasters.forecaster import Forecaster, TransformerForecaster
    from picid.model.forecasters.lstm_model import LSTM_Forecaster
    from picid.model.forecasters.patchtst_model import PatchTST_Forecaster
    from picid.model.forecasters.spacetimeformer_model import Spacetimeformer_Forecaster
    from picid.model.forecasters.tide_model import TiDE_Forecaster
    from picid.model.forecasters.timeseries_transformer_model import (
        Timeseries_Transformer_Forecaster,
    )
