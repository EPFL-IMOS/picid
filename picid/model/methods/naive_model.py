"""Legacy import path for the window-average baseline."""

from picid.model.estimators.window_average.model import WindowAverageBaseline

NaiveModelBaseline = WindowAverageBaseline
__all__ = ["NaiveModelBaseline"]
