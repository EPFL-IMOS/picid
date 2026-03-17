from pydantic import Field

from .base import AbsModelConfig


class MeanConfig(AbsModelConfig):
    config_name : str = Field('mean', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('picid.model.wrappers.naive_model_wrapper.NaiveModelWrapper',
                                 frozen=True, serialization_alias='_target_')

    pred_len: int = Field(ge=1)
    label_len: int = Field(ge=1)
    seq_len: int = Field(ge=1)
    window_size_to_average: int = Field(ge=1)

    features_mode: str
