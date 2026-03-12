from pydantic import Field

from .base import AbsModelConfig


class CNN1DConfig(AbsModelConfig):
    config_name : str = Field('cnn_1d', frozen=True)
    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field('picid.model.wrappers.cnn1d_wrapper.CNN1D_Wrapper',
                                frozen=True, serialization_alias='_target_')

    input_channels: int = Field(ge=1)
    seq_len: int = Field(ge=1)

    kernels: int = Field(ge=1)
    strides: int = Field(ge=1)
    dilations: int = Field(ge=0)
    latent_dim: int = Field(ge=1)
    dropout_prob: float = Field(ge=0.1, le=1.0)

    output_channels: int = Field(ge=1)
