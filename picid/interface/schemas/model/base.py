from abc import ABC

from pydantic import BaseModel


class AbsModelConfig(BaseModel, ABC):
    model_class : str
    config_name : str