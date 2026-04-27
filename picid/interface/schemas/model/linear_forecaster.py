from typing import Optional

from pydantic import Field

from .base import AbsModelConfig


class LinearForecasterConfig(AbsModelConfig):
    """Config for the linear forecasting baseline.

    A lightweight linear model for time-series forecasting. Supports optional
    reversible instance normalisation and seasonal decomposition.

    Parameters
    ----------
    linear_window : int
        Look-back window size for the linear layer. ``0`` uses the full
        context window. Default ``0``.
    linear_shared_weights : bool
        Share weights across all feature dimensions. Default ``False``.
    use_revin : bool
        Enable reversible instance normalisation. Default ``False``.
    use_seasonal_decomp : bool
        Enable seasonal decomposition. Default ``False``.
    context_points : int or None
        Context length override. Inferred from task definition when ``None``.
    d_x, d_yt, d_yc : int or None
        Dimension overrides; inferred from data when ``None``.
    """

    config_name: str = Field("linear_forecaster", frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field(
        "picid.model.forecasters.linear_model.linear_model.Linear_Forecaster",
        frozen=True,
        serialization_alias="_target_",
    )

    d_x: Optional[int] = Field(None)
    d_yt: Optional[int] = Field(None)
    d_yc: Optional[int] = Field(None)
    context_points: Optional[int] = Field(None)

    linear_window: int = Field(0, ge=0)

    linear_shared_weights: bool = False
    use_revin: bool = False
    use_seasonal_decomp: bool = False
