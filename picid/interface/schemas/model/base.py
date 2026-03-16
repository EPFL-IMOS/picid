from abc import ABC

from pydantic import BaseModel, Field


class AbsModelConfig(BaseModel, ABC):
    model_class : str = Field(frozen=True)
    config_name : str = Field(frozen=True)