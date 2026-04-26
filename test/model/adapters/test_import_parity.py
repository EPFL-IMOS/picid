"""Import tests for the canonical adapter namespace."""

from picid.model.adapters import (
    AbstractFeedForwardTrainingWrapper as PackageFeedForwardTrainingWrapper,
)
from picid.model.adapters import AbstractFeedForwardWrapper as PackageFeedForwardWrapper
from picid.model.adapters import AbstractFitPredictWrapper as PackageFitPredictWrapper
from picid.model.adapters.base import AbstractFeedForwardWrapper
from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper
from picid.model.adapters.base import AbstractFitPredictWrapper


def test_adapter_package_re_exports_match_base_module():
    assert PackageFitPredictWrapper is AbstractFitPredictWrapper
    assert PackageFeedForwardWrapper is AbstractFeedForwardWrapper
    assert PackageFeedForwardTrainingWrapper is AbstractFeedForwardTrainingWrapper
