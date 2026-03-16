from pydantic import Field

from .base import AbsModelConfig


class LinearRegressionConfig(AbsModelConfig):
    config_name : str = Field('linear_regression', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('picid.model.wrappers.statistical_models_wrapper.StatisticalBaselineWrapper',
                                 frozen=True, serialization_alias='_target_')

    pred_len: int = Field(ge=1)
    label_len: int = Field(ge=1)
    seq_len: int = Field(ge=1)

    input_channels: int = Field(ge=1)
    num_targets: int = Field(ge=1)

    model_type: str
