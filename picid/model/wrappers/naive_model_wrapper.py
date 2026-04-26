"""Legacy import path for the window-average wrapper."""

from picid.model.estimators.window_average.wrapper import WindowAverageWrapper

NaiveModelWrapper = WindowAverageWrapper
__all__ = ["NaiveModelWrapper"]
