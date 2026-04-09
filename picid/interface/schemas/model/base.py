from abc import ABC

from pydantic import BaseModel, Field


class AbsModelConfig(BaseModel, ABC):
    """Abstract base for all model configs. Not instantiated directly.

    Each concrete subclass maps to a YAML file under
    ``configs/model_configs/<task>/<config_name>.yaml``. The ``config_name``
    field selects that file, and any fields you set on the subclass override
    the YAML defaults before the model is instantiated by Hydra.
    """

    model_class : str = Field(frozen=True)
    config_name : str = Field(frozen=True)