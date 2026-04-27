"""Window-average estimator shim."""

from picid.model.estimators.window_average.model import WindowAverageBaseline
from picid.model.estimators.window_average.wrapper import WindowAverageWrapper

__all__ = ["WindowAverageBaseline", "WindowAverageWrapper"]
