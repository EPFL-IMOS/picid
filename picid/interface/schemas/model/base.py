from abc import ABC
from typing import List, Any

from pydantic import BaseModel, Field


class AbsModelConfig(BaseModel, ABC):
    model_class : str = Field(frozen=True)
    config_name : str = Field(frozen=True)

    # defaults: List[Any] = Field(default_factory=lambda: [
    #     {"override model": "model_cfg"}
    # ])