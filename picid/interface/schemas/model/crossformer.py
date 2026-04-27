from typing import Optional

from pydantic import Field

from .base import AbsModelConfig


class CrossformerConfig(AbsModelConfig):
    """Config for the Crossformer Transformer model.

    A Transformer-based architecture designed for multivariate time-series
    forecasting. Input/output dimensions are inferred from the data when left
    as ``None``.

    Parameters
    ----------
    dropout : float
        Dropout rate. Must be in ``[0.1, 1.0]``. Required.
    d_model : int
        Model embedding dimension. Default ``128``.
    d_ff : int
        Feed-forward layer dimension. Default ``128``.
    n_heads : int
        Number of attention heads. Default ``4``.
    e_layers : int
        Number of encoder layers. Default ``3``.
    seg_len : int
        Segment length for cross-dimension attention. Default ``6``.
    win_size : int
        Window size for hierarchical attention. Default ``2``.
    factor : int
        Attention sampling factor. Default ``10``.
    use_revin : bool
        Enable reversible instance normalisation. Default ``False``.
    use_seasonal_decomp : bool
        Enable seasonal decomposition pre-processing. Default ``False``.
    d_x, d_yt, d_yc, ts_in, ts_out : int or None
        Dimension overrides; inferred from data when ``None``.

    Examples
    --------
    >>> model = CrossformerConfig(dropout=0.1)
    >>> model = CrossformerConfig(dropout=0.2, d_model=256, n_heads=8)
    """

    config_name: str = Field("lstm", frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field(
        "picid.model.forecasters.crossformer_model.Crossformer_Forecaster",
        frozen=True,
        serialization_alias="_target_",
    )

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
    decoder_embedding: str = "DSW"

    baseline: bool = False
    use_revin: bool = False
    use_seasonal_decomp: bool = False
