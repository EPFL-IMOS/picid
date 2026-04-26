"""Window-average wrapper shim."""

from picid.model.wrappers.naive_model_wrapper import (
    NaiveModelWrapper as WindowAverageWrapper,
)

__all__ = ["WindowAverageWrapper"]
