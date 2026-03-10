from abc import ABC
from typing import Annotated

from pydantic import BaseModel, Field

TargetField = Annotated[str, Field(frozen=True, serialization_alias='_target_')]

class AbsHookConfig(BaseModel, ABC):
    model_class : TargetField


class ReconstructionPlotHookConfig(AbsHookConfig):
    model_class : TargetField = 'picid.evaluator.hooks.reconstruction_plot.ReconstructionPlotHook'

class SavePredictionsHookConfig(AbsHookConfig):
    model_class : TargetField = 'picid.evaluator.hooks.save_predictions.SavePredictionsHook'
    dims: list[str]

class UnitTrendPlotHookConfig(AbsHookConfig):
    model_class : TargetField = 'picid.evaluator.hooks.unit_trend_plot.UnitTrendPlotHook'
    log_every_n_epochs: int = 10
    enable_subsampling: bool = True
    subsample_threshold: int = 2000
    subsample_factor: int = 10
