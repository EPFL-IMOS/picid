"""Canonical namespace for the LSTM forecaster family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["LSTM_Forecaster"]


def __getattr__(name):
    if name != "LSTM_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("picid.model.forecasters.lstm_model.lstm_model")
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.lstm_model.lstm_model import LSTM_Forecaster
