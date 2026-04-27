from pydantic import Field

from .base import AbsModelConfig


class LinearRegressionConfig(AbsModelConfig):
    """Config for a statistical linear-regression baseline.

    A non-neural baseline that fits a linear model on windowed features.
    All architecture fields are required.

    Parameters
    ----------
    pred_len : int
        Prediction horizon length. Required.
    label_len : int
        Label overlap length. Required.
    seq_len : int
        Input window length. Must match the task definition. Required.
    input_channels : int
        Number of input features. Required.
    num_targets : int
        Number of output targets. Required.
    model_type : str
        Regression model variant (e.g. ``"linear"``). Required.
    """

    config_name : str = Field('linear_regression', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('picid.model.estimators.statistical.wrapper.StatisticalBaselineWrapper',
                                 frozen=True, serialization_alias='_target_')

    pred_len: int = Field(ge=1)
    label_len: int = Field(ge=1)
    seq_len: int = Field(ge=1)

    input_channels: int = Field(ge=1)
    num_targets: int = Field(ge=1)

    model_type: str
