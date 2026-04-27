"""Canonical namespace for the Spacetimeformer forecaster family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Spacetimeformer_Forecaster"]


def __getattr__(name):
    if name != "Spacetimeformer_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(
        "picid.model.forecasters.spacetimeformer_model.spacetimeformer_model"
    )
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.spacetimeformer_model.spacetimeformer_model import (
        Spacetimeformer_Forecaster,
    )
