"""Canonical namespace for the PatchTST forecaster family."""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["PatchTST_Forecaster"]


def __getattr__(name):
    if name != "PatchTST_Forecaster":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(
        "picid.model.forecasters.patchtst_model.thuml_patchtst_model"
    )
    return getattr(module, name)


if TYPE_CHECKING:
    from picid.model.forecasters.patchtst_model.thuml_patchtst_model import (
        PatchTST_Forecaster,
    )
