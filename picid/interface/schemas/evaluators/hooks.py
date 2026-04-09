from abc import ABC
from typing import Annotated

from pydantic import BaseModel, Field

TargetField = Annotated[str, Field(frozen=True, serialization_alias='_target_')]

__all__ = ["AbsHookConfig", "ReconstructionPlotHookConfig", "SavePredictionsHookConfig",
           "UnitTrendPlotHookConfig"]

class AbsHookConfig(BaseModel, ABC):
    """Abstract base for evaluator hook configs. Not instantiated directly."""

    model_class : TargetField


class ReconstructionPlotHookConfig(AbsHookConfig):
    """Hook that logs reconstruction plots after evaluation."""

    model_class : TargetField = 'picid.evaluator.hooks.reconstruction_plot.ReconstructionPlotHook'

class SavePredictionsHookConfig(AbsHookConfig):
    """Hook that saves raw model predictions to disk after evaluation.

    Parameters
    ----------
    dims : list[str]
        Names for the output tensor dimensions (e.g. ``["sample", "time", "feature"]``).
    """

    model_class : TargetField = 'picid.evaluator.hooks.save_predictions.SavePredictionsHook'
    dims: list[str]

class UnitTrendPlotHookConfig(AbsHookConfig):
    """Hook that logs per-unit prediction trend plots during evaluation.

    Parameters
    ----------
    log_every_n_epochs : int
        Plot every N epochs. Default ``10``.
    enable_subsampling : bool
        Subsample long series before plotting. Default ``True``.
    subsample_threshold : int
        Series longer than this (in samples) are subsampled. Default ``2000``.
    subsample_factor : int
        Keep every N-th sample when subsampling. Default ``10``.
    """

    model_class : TargetField = 'picid.evaluator.hooks.unit_trend_plot.UnitTrendPlotHook'
    log_every_n_epochs: int = 10
    enable_subsampling: bool = True
    subsample_threshold: int = 2000
    subsample_factor: int = 10
