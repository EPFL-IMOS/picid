from pydantic import Field

from .base import AbsModelConfig


class MeanConfig(AbsModelConfig):
    """Config for the windowed-mean forecasting baseline.

    Forecasts by averaging the last ``window_size_to_average`` observed values.
    All fields are required.

    Parameters
    ----------
    pred_len : int
        Prediction horizon. Required.
    label_len : int
        Label overlap. Required.
    seq_len : int
        Input window length. Must match the task definition. Required.
    window_size_to_average : int
        Number of past observations to average. Required.
    features_mode : str
        Feature mode string (e.g. ``"M"`` for multivariate). Required.
    """

    config_name : str = Field('mean', frozen=True)

    # lightning_model: bool = Field(True, frozen=True)
    model_class : str = Field('picid.model.estimators.window_average.wrapper.WindowAverageWrapper',
                                 frozen=True, serialization_alias='_target_')

    pred_len: int = Field(ge=1)
    label_len: int = Field(ge=1)
    seq_len: int = Field(ge=1)
    window_size_to_average: int = Field(ge=1)

    features_mode: str
