"""Compatibility re-export for the legacy forecaster base classes."""

from picid.model.forecasters.forecaster import Forecaster, TransformerForecaster

__all__ = ["Forecaster", "TransformerForecaster"]
