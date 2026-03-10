from abc import ABC
from typing import Annotated

from pydantic import BaseModel, Field, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from picid.evaluator.hooks import BaseEvalHook
from picid.interface.schemas.evaluators.hooks import AbsHookConfig, SavePredictionsHookConfig, UnitTrendPlotHookConfig

TargetField = Annotated[str, Field(frozen=True, serialization_alias='_target_')]

class AbsEvalConfig(BaseModel, ABC):
    model_class : TargetField
    # config_name : str = Field(frozen=True)

    metric_names : list[str]

    save_predictions: bool = False
    apply_inverse_scaling: bool = False

    task_mode: str = '${datasource.task_mode}'
    paths: str = None

    hooks : list[AbsHookConfig] | None = []


class DefaultEvaluatorConfig(AbsEvalConfig):
    model_class : TargetField = 'picid.evaluator.default.DefaultEvaluator'
    metric_names : list[str] = ["mae", "mse", "rmse"]


class ClassificationEvaluatorConfig(AbsEvalConfig):
    model_class : TargetField = 'picid.evaluator.classification.ClassificationEvaluator'
    metric_names : list[str] = ["f1", "accuracy", "precision", "recall", "auroc"]

    num_classes: int | str = '${task_definition.num_classes}'

    hooks: list[AbsHookConfig] = [SavePredictionsHookConfig(dims=["sample", "time", "feature"])]


class RulEvaluatorConfig(DefaultEvaluatorConfig):
    metric_names : list[str] = ["mae", "mse", "rmse", "nasa_score"]


class PerUnitEvaluatorConfig(AbsEvalConfig):
    model_class : TargetField = 'picid.evaluator.multiunit.MultiUnitEvaluator'
    metric_names : list[str] = ["phm_score", "mae", "mse", "rmse"]

    log_image: bool = False
    inverse_transform_name: str | None = None
    hooks: list[AbsHookConfig] = [SavePredictionsHookConfig(dims=["sample", "time", "feature"]),
                                  UnitTrendPlotHookConfig()]


class ForecastingEvaluatorConfig(AbsEvalConfig):
    model_class : TargetField = 'picid.evaluator.forecasting.ForecastingEvaluator'
    metric_names : list[str] = ["mae", "mse"]

    log_image: bool = False
    inverse_transform_name: str | None = None
    apply_inverse_scaling : bool = True

    model_seq_len: int | str = '${task_definition.seq_len}'
    model_label_len: int | str = '${task_definition.label_len}'
    model_pred_len: int | str = '${task_definition.pred_len}'

    target_dim_position: int | None = None
    hooks: list[AbsHookConfig] = [SavePredictionsHookConfig(dims=["sample", "time", "feature"])]


print(DefaultEvaluatorConfig().model_dump(by_alias=True))