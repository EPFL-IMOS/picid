"""Canonical namespace for the TiDE forecaster family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["TiDE_Forecaster"]


def __getattr__(name):
    if name != "TiDE_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("picid.model.forecasters.tide_model.tide_model")
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.tide_model.tide_model import TiDE_Forecaster
