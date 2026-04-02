from abc import ABC
from typing import Literal

from pydantic import Field, model_validator, BaseModel, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler


class BaseTaskDefinition(BaseModel, ABC):
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
    config_name: str = Field("prognostics", frozen=True)

    task_type: Literal["ahrul", "rul"]
    pred_len: int = 0


class Forecasting(BaseTaskDefinition):
    config_name: str = Field("forecasting", frozen=True)

    task_type: str = Field("forecasting", frozen=True)
    pred_offset: int = Field(0)

    input_tensors: list[str] = ["features", "time_features", "target"]
