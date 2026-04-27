"""Canonical namespace for the linear forecaster family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Linear_Forecaster"]


def __getattr__(name):
    if name != "Linear_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("picid.model.forecasters.linear_model.linear_model")
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.linear_model.linear_model import Linear_Forecaster
