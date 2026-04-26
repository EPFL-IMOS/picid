"""Compatibility re-export for the shared wrapper base classes."""

from picid.model.wrappers.base import AbstractFeedForwardTrainingWrapper
from picid.model.wrappers.base import AbstractFeedForwardWrapper
from picid.model.wrappers.base import AbstractFitPredictWrapper

__all__ = [
    "AbstractFeedForwardTrainingWrapper",
    "AbstractFeedForwardWrapper",
    "AbstractFitPredictWrapper",
]
