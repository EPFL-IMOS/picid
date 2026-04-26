from pydantic import Field

from .base import AbsModelConfig


class MLPConfig(AbsModelConfig):
    """Config for the multi-layer perceptron (MLP) model.

    Suitable for prognostics and regression tasks. Flattens the input window
    before processing, so ``input_channels`` must equal
    ``seq_len × n_features``.

    Parameters
    ----------
    input_channels : int
        Total number of input units after flattening. Required.
    num_targets : int
        Number of output units. Required.
    hidden_dim : int
        Width of hidden layers. Default ``64``.

    Examples
    --------
    >>> model = MLPConfig(input_channels=160, num_targets=1)
    """

    config_name : str = Field('mlp', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field('picid.model.estimators.mlp.wrapper.MLPWrapper',
                                frozen=True, serialization_alias='_target_')

    input_channels: int = Field(ge=1)
    num_targets: int = Field(ge=1)
    hidden_dim: int = Field(64, ge=1)
