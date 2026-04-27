"""Canonical namespace for Spacetimeformer neural helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Spacetimeformer"]


def __getattr__(name):
    if name != "Spacetimeformer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("picid.model.forecasters.spacetimeformer_model.nn.model")
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.spacetimeformer_model.nn.model import Spacetimeformer
