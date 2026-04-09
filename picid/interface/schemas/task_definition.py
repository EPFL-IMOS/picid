from abc import ABC
from typing import Literal

from pydantic import Field, model_validator, BaseModel, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler


class BaseTaskDefinition(BaseModel, ABC):
    """Abstract base for all task-definition configs.

    A task definition controls how raw time-series data is windowed and what
    the model is trained to predict. Subclass with ``Prognostic`` or
    ``Forecasting`` — do not instantiate directly.

    Parameters
    ----------
    seq_len : int
        Number of input time steps (window length). Default ``16``.
    label_len : int
        Overlap between the input window and the prediction horizon, used by
        some Transformer architectures. Default ``0``.
    pred_len : int
        Number of future time steps to predict. Set to ``0`` for prognostics
        tasks. Default ``1``.
    stride : int
        Sliding-window step size during evaluation. Default ``1``.
    stride_train : int
        Sliding-window step size during training. Default ``1``.
    subset_ratio : float
        Fraction of training windows to keep (useful for fast prototyping).
        Must be in ``(0, 1]``. Default ``1.0``.
    padding_left_flag : bool
        Pad the left edge of the series so every sample has a full window.
        Default ``False``.
    task_type : str
        Key used to retrieve the target tensor from the data dictionary
        (e.g. ``"rul"``, ``"ahrul"``, ``"forecasting"``). Must be set by
        subclasses.
    target_metric : str
        Metric used for model checkpointing. Default ``"val/loss"``.
    target_metric_mode : str
        Whether to ``"min"`` or ``"max"`` the ``target_metric``.
        Default ``"min"``.
    """

    config_name : str

    seq_len: int = Field(16, ge=0)  # window size
    label_len: int = Field(0, ge=0)  # overlap size
    pred_len: int = Field(1, ge=0)  # prediction size

    stride: int = Field(1, ge=1)
    stride_train: int = Field(1, ge=1)
    subset_ratio: float = Field(1.0, ge=0, le=1)
    subset_seed: int = Field(0)

    padding_left_flag: bool = False
    input_tensors: list[str] = Field(["features"], exclude=True)

    task_type: str
    target_metric: str = "val/loss"
    target_metric_mode: str = "min"

    @model_validator(mode="after")
    def propagate_tag(self) -> "BaseTaskDefinition":
        if self.task_type not in self.input_tensors and self.task_type != "forecasting":
            self.input_tensors.append(self.task_type)

        return self

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict:
        serialized = handler(self)

        serialized["model"] = {
            "data_requirements": {"input_tensors": self.input_tensors}
        }

        return serialized


class Prognostic(BaseTaskDefinition):
    """Task definition for prognostics (RUL / AHRUL prediction).

    Use this when the model should predict the Remaining Useful Life (RUL) or
    Adapted Horizon RUL (AHRUL) of a component. ``pred_len`` is fixed at ``0``
    because prognostic models output a scalar health indicator rather than a
    multi-step horizon.

    Parameters
    ----------
    task_type : {"rul", "ahrul"}
        Key used to retrieve the target tensor from the data dictionary.
        Must match the ``task_mode`` of your datasource.

    Examples
    --------
    >>> task = Prognostic(task_type="rul")
    >>> task = Prognostic(task_type="rul", seq_len=32, stride=2)
    """

    config_name: str = Field("prognostics", frozen=True)

    task_type: Literal["ahrul", "rul"]
    pred_len: int = 0


class Forecasting(BaseTaskDefinition):
    """Task definition for multi-step time-series forecasting.

    Use this when the model should predict ``pred_len`` future time steps from
    a context window of ``seq_len`` steps. The dataloader automatically
    provides ``"features"``, ``"time_features"``, and ``"target"`` tensors.

    Parameters
    ----------
    seq_len : int
        Context window length. Default ``16``.
    label_len : int
        Overlap between the input window and the forecast horizon. Required by
        some encoder-decoder Transformers. Default ``0``.
    pred_len : int
        Number of future steps to forecast. Default ``1``.
    pred_offset : int
        Time steps to skip between the end of the input window and the start
        of the forecast horizon. Default ``0``.

    Examples
    --------
    >>> task = Forecasting(seq_len=96, pred_len=24)
    >>> task = Forecasting(seq_len=336, label_len=48, pred_len=96)
    """

    config_name: str = Field("forecasting", frozen=True)

    task_type: str = Field("forecasting", frozen=True)
    pred_offset: int = Field(0)

    input_tensors: list[str] = ["features", "time_features", "target"]
