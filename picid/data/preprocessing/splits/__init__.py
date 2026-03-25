"""Legacy splitter import path.

This package remains as a compatibility layer while the canonical public
import path moves to :mod:`picid.data.split_strategies`.
"""

from picid.data.split_strategies import (
    BySourceSplitter,
    SourceSplitter,
    TimeSeriesSplitter,
    TimeSplitter,
    TimeStampSplitter,
    ValueWarning,
)

__all__ = [
    "BySourceSplitter",
    "SourceSplitter",
    "TimeSeriesSplitter",
    "TimeSplitter",
    "TimeStampSplitter",
    "ValueWarning",
]
