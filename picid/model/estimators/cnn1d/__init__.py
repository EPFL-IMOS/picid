"""CNN1D estimator shim."""

from picid.model.estimators.cnn1d.model import EncoderModel
from picid.model.estimators.cnn1d.wrapper import CNN1D_Wrapper

__all__ = ["CNN1D_Wrapper", "EncoderModel"]
