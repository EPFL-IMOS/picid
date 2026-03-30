"""Shared enums for split-aware data containers."""

from enum import StrEnum


class SplitUnitCardinality(StrEnum):
    """Describe how many units each populated split contains."""

    EMPTY = "empty"
    SINGLE_UNIT_PER_SPLIT = "single_unit_per_split"
    MULTI_UNIT_PER_SPLIT = "multi_unit_per_split"
    MIXED = "mixed"


class SplitViewPolicy(StrEnum):
    """Control how split-first views are exported from a container."""

    KEEP_UNIT_LISTS = "keep_unit_lists"
    UNWRAP_SINGLETONS = "unwrap_singletons"
