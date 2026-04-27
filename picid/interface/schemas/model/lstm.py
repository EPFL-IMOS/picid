from typing import Optional

from pydantic import Field

from .base import AbsModelConfig


class LSTMConfig(AbsModelConfig):
    """Config for the LSTM-based forecaster/prognostics model.

    Suitable for both RUL prognostics and time-series forecasting tasks.
    Input/output dimensions (``d_x``, ``d_yt``, ``d_yc``) are inferred from
    the data when left as ``None``.

    Parameters
    ----------
    hidden_dim : int
        Hidden state size per LSTM layer. Default ``32``.
    n_layers : int
        Number of stacked LSTM layers. Default ``2``.
    d_x : int or None
        Input feature dimension. Inferred from data when ``None``.
    d_yt : int or None
        Target dimension. Inferred from data when ``None``.
    d_yc : int or None
        Context dimension. Inferred from data when ``None``.

    Examples
    --------
    >>> model = LSTMConfig()
    >>> model = LSTMConfig(n_layers=8, hidden_dim=64)
    """

    config_name: str = Field("lstm", frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class: str = Field(
        "picid.model.forecasters.lstm_model.LSTM_Forecaster",
        frozen=True,
        serialization_alias="_target_",
    )

    d_x: Optional[int] = Field(None)
    d_yt: Optional[int] = Field(None)
    d_yc: Optional[int] = Field(None)

    hidden_dim: int = Field(32, ge=1)
    n_layers: int = Field(2, ge=1)
