"""Canonical namespace for the timeseries transformer family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Timeseries_Transformer_Forecaster"]


def __getattr__(name):
    if name != "Timeseries_Transformer_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(
        "picid.model.forecasters.timeseries_transformer_model.timeseries_transformer_model"
    )
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.timeseries_transformer_model.timeseries_transformer_model import (
        Timeseries_Transformer_Forecaster,
    )
