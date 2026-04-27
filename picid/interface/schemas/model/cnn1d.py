from pydantic import Field

from .base import AbsModelConfig


class CNN1DConfig(AbsModelConfig):
    """Config for the 1-D convolutional encoder model.

    Suitable for prognostics and feature-extraction tasks. All architecture
    fields are required because the model cannot infer them from the data.

    Parameters
    ----------
    input_channels : int
        Number of input feature channels. Required.
    seq_len : int
        Length of the input window. Must match the task definition. Required.
    kernels : int
        Convolutional kernel size. Required.
    strides : int
        Convolutional stride. Required.
    dilations : int
        Dilation factor. Required.
    latent_dim : int
        Dimensionality of the encoder output. Required.
    dropout_prob : float
        Dropout probability. Must be in ``[0.1, 1.0]``. Required.
    output_channels : int
        Number of output channels from the final conv layer. Required.
    """

    config_name : str = Field('cnn_1d', frozen=True)
    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field('picid.model.estimators.cnn1d.wrapper.CNN1D_Wrapper',
                                frozen=True, serialization_alias='_target_')

    input_channels: int = Field(ge=1)
    seq_len: int = Field(ge=1)

    kernels: int = Field(ge=1)
    strides: int = Field(ge=1)
    dilations: int = Field(ge=0)
    latent_dim: int = Field(ge=1)
    dropout_prob: float = Field(ge=0.1, le=1.0)

    output_channels: int = Field(ge=1)
