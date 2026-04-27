"""Canonical namespace for the Crossformer forecaster family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Crossformer_Forecaster"]


def __getattr__(name):
    if name != "Crossformer_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(
        "picid.model.forecasters.crossformer_model.crossformer_model"
    )
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.crossformer_model.crossformer_model import (
        Crossformer_Forecaster,
    )
