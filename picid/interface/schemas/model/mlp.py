from pydantic import Field

from .base import AbsModelConfig


class MLPConfig(AbsModelConfig):
    config_name : str = Field('mlp', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field('picid.model.wrappers.mlp_wrapper.MLPWrapper',
                                frozen=True, serialization_alias='_target_')

    input_channels: int = Field(ge=1)
    num_targets: int = Field(ge=1)
    hidden_dim: int = Field(64, ge=1)
