"""Legacy shim for the canonical CNN1D wrapper."""

from picid.model.estimators.cnn1d.wrapper import CNN1D_Wrapper
from picid.model.estimators.cnn1d.wrapper import _calculate_receptive_field

__all__ = ["CNN1D_Wrapper", "_calculate_receptive_field"]
