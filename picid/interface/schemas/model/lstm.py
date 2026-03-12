from typing import Optional

from pydantic import Field

from .base import AbsModelConfig


class LSTMConfig(AbsModelConfig):
    config_name : str = Field('lstm', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('picid.baselines.lstm_model.LSTM_Forecaster',
                                 frozen=True, serialization_alias='_target_')

    d_x: Optional[int] = Field(None)
    d_yt: Optional[int] = Field(None)
    d_yc: Optional[int] = Field(None)

    hidden_dim: int = Field(32, ge=1)
    n_layers: int = Field(2, ge=1)
