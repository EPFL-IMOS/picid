"""Legacy compatibility shim for shared wrapper base classes."""

from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper
from picid.model.adapters.base import AbstractFeedForwardWrapper
from picid.model.adapters.base import AbstractFitPredictWrapper

__all__ = [
    "AbstractFeedForwardTrainingWrapper",
    "AbstractFeedForwardWrapper",
    "AbstractFitPredictWrapper",
]
