"""Import parity tests for the canonical adapter namespace."""

from picid.model.adapters import (
    AbstractFeedForwardTrainingWrapper as PackageFeedForwardTrainingWrapper,
)
from picid.model.adapters import AbstractFeedForwardWrapper as PackageFeedForwardWrapper
from picid.model.adapters import AbstractFitPredictWrapper as PackageFitPredictWrapper
from picid.model.adapters.base import AbstractFeedForwardWrapper
from picid.model.adapters.base import AbstractFeedForwardTrainingWrapper
from picid.model.adapters.base import AbstractFitPredictWrapper
from picid.model.wrappers.base import (
    AbstractFeedForwardWrapper as LegacyFeedForwardWrapper,
)
from picid.model.wrappers.base import (
    AbstractFeedForwardTrainingWrapper as LegacyFeedForwardTrainingWrapper,
)
from picid.model.wrappers.base import (
    AbstractFitPredictWrapper as LegacyFitPredictWrapper,
)


def test_adapter_base_imports_match_legacy_wrapper_base():
    assert AbstractFitPredictWrapper is LegacyFitPredictWrapper
    assert AbstractFeedForwardWrapper is LegacyFeedForwardWrapper
    assert AbstractFeedForwardTrainingWrapper is LegacyFeedForwardTrainingWrapper


def test_adapter_package_re_exports_match_base_module():
    assert PackageFitPredictWrapper is AbstractFitPredictWrapper
    assert PackageFeedForwardWrapper is AbstractFeedForwardWrapper
    assert PackageFeedForwardTrainingWrapper is AbstractFeedForwardTrainingWrapper
