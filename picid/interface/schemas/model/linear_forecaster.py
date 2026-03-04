from typing import Optional

from pydantic import Field

from .base import AbsModelConfig


class LinearForecasterConfig(AbsModelConfig):
    config_name : str = Field('linear_forecaster', frozen=True)

    lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('lmetk.baselines.linear_model.linear_model.Linear_Forecaster',
                                 frozen=True, serialization_alias='_target_')

    d_x: Optional[int] = Field(None)
    d_yt: Optional[int] = Field(None)
    d_yc: Optional[int] = Field(None)
    context_points: Optional[int] = Field(None)

    linear_window: int = Field(0, ge=0)

    linear_shared_weights: bool = False
    use_revin: bool = False
    use_seasonal_decomp: bool = False