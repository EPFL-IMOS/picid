from typing import Optional

from pydantic import Field

from .base import AbsModelConfig


class CrossformerConfig(AbsModelConfig):
    config_name : str = Field('lstm', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('picid.baselines.crossformer_model.Crossformer_Forecaster',
                                 frozen=True, serialization_alias='_target_')

    d_x: Optional[int] = Field(None)
    d_yt: Optional[int] = Field(None)
    d_yc: Optional[int] = Field(None)
    ts_in: Optional[int] = Field(None)
    ts_out: Optional[int] = Field(None)

    d_model: int = Field(128, ge=1)
    d_ff: int = Field(128, ge=1)

    seg_len: int = Field(6, ge=1)
    win_size: int = Field(2, ge=1)
    factor: int = Field(10, ge=1)
    n_heads: int = Field(4, ge=1)
    e_layers: int = Field(3, ge=1)
    dropout: float = Field(ge=0.1, le=1.0)
    decoder_embedding: str = 'DSW'

    baseline: bool = False
    use_revin: bool = False
    use_seasonal_decomp: bool = False