from abc import ABC
from typing import Annotated

from pydantic import BaseModel, Field

from picid.interface.schemas.evaluators.hooks import AbsHookConfig, SavePredictionsHookConfig, UnitTrendPlotHookConfig

__all__ = ["AbsEvalConfig", "DefaultEvaluatorConfig", "ClassificationEvaluatorConfig",
           "RulEvaluatorConfig", "PerUnitEvaluatorConfig", "ForecastingEvaluatorConfig"]

TargetField = Annotated[str, Field(frozen=True, serialization_alias='_target_')]

class AbsEvalConfig(BaseModel, ABC):
    """Abstract base for all evaluator configs. Not instantiated directly.

    Parameters
    ----------
    metric_names : list[str]
        Names of metrics to compute (e.g. ``["mae", "mse", "rmse"]``).
    save_predictions : bool
        Persist raw model predictions to disk after evaluation. Default ``False``.
    apply_inverse_scaling : bool
        Undo scaling transforms before computing metrics. Requires
        ``inverse_transform_name`` to be set (or a compatible transform to be
        auto-detected). Default ``False``.
    hooks : list[AbsHookConfig] or None
        Post-evaluation hooks (plots, file exports, etc.). Default ``[]``.
    """

    model_class : TargetField

    metric_names : list[str]

    save_predictions: bool = False
    apply_inverse_scaling: bool = False

    task_mode: str = '${datasource.task_mode}'
    paths: str = None

    hooks : list[AbsHookConfig] | None = []


class DefaultEvaluatorConfig(AbsEvalConfig):
    """General-purpose regression evaluator.

    Computes ``mae``, ``mse``, and ``rmse`` by default. Suitable for any
    regression task when no task-specific evaluator is needed.
    """

    model_class : TargetField = 'picid.evaluator.default.DefaultEvaluator'
    metric_names : list[str] = ["mae", "mse", "rmse"]


class ClassificationEvaluatorConfig(AbsEvalConfig):
    """Evaluator for classification tasks.

    Computes ``f1``, ``accuracy``, ``precision``, ``recall``, and ``auroc``
    by default. The ``num_classes`` field is automatically resolved from the
    task definition config when left at its default.

    Parameters
    ----------
    num_classes : int or str
        Number of target classes. Defaults to the OmegaConf interpolation
        ``"${task_definition.num_classes}"`` which is resolved at runtime.
    """

    model_class : TargetField = 'picid.evaluator.classification.ClassificationEvaluator'
    metric_names : list[str] = ["f1", "accuracy", "precision", "recall", "auroc"]

    num_classes: int | str = '${task_definition.num_classes}'

    hooks: list[AbsHookConfig] = [SavePredictionsHookConfig(dims=["sample", "time", "feature"])]


class RulEvaluatorConfig(DefaultEvaluatorConfig):
    """Evaluator for RUL (Remaining Useful Life) prognostics tasks.

    Extends ``DefaultEvaluatorConfig`` by adding the NASA prognostic score
    (``nasa_score``) to the default metric set.
    """

    metric_names : list[str] = ["mae", "mse", "rmse", "nasa_score"]


class PerUnitEvaluatorConfig(AbsEvalConfig):
    """Evaluator that computes metrics independently for each unit (machine/subject).

    Useful when the test set contains multiple distinct machines and you want
    per-unit breakdowns rather than a global aggregate. Includes a trend plot
    hook by default.

    Parameters
    ----------
    log_image : bool
        Log per-unit prediction trend plots to the logger. Default ``False``.
    inverse_transform_name : str or None
        Name of the ``DataTransform`` whose inverse should be applied before
        computing metrics. Must match the ``transform_name`` of a transform
        that implements ``InverseTransformMixin``. Default ``None``.
    """

    model_class : TargetField = 'picid.evaluator.multiunit.MultiUnitEvaluator'
    metric_names : list[str] = ["phm_score", "mae", "mse", "rmse"]

    log_image: bool = False
    inverse_transform_name: str | None = None
    hooks: list[AbsHookConfig] = [SavePredictionsHookConfig(dims=["sample", "time", "feature"]),
                                  UnitTrendPlotHookConfig()]


class ForecastingEvaluatorConfig(AbsEvalConfig):
    """Evaluator for multi-step forecasting tasks.

    Handles window alignment between model output and ground truth using the
    sequence length parameters, which are resolved from the task definition
    config at runtime. Applies inverse scaling by default.

    Parameters
    ----------
    log_image : bool
        Log forecast plots to the logger. Default ``False``.
    inverse_transform_name : str or None
        Name of the ``DataTransform`` to invert before computing metrics.
        Default ``None`` (auto-detected from transforms list).
    apply_inverse_scaling : bool
        Undo scaling before computing metrics. Default ``True``.
    target_dim_position : int or None
        Index of the target feature dimension in the forecast output, when the
        model predicts multiple features. Default ``None``.
    """

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
