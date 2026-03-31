"""Shared datasource contracts and lifecycle enums.

This module groups the structural datasource protocols together with the small
set of enums and normalization helpers that define the loader contract.
Keeping them together makes the supported base-layer API easier to review than
spreading the contract across multiple tiny modules.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from picid.data.data_objects import DatasetContainer, SplitDatasetContainer


class LoaderState(StrEnum):
    """
    Define the Loader State helper.

    """

    INITIALIZED = "initialized"
    LOADED = "loaded"
    SPLIT = "split"


class SplitMode(StrEnum):
    """
    Define the Split Mode helper.

    """

    WITHIN_UNITS = "within_units"
    BETWEEN_UNITS = "between_units"

@runtime_checkable
class DatasourceProtocol(Protocol):
    """
    Define the Datasource Protocol helper.

    """

    data_name: str
    task_mode: str

    def load_data(self) -> None: ...

    def split_data(self) -> None: ...

    def get_data(self) -> DatasetContainer | SplitDatasetContainer: ...

    def get_meta_data(self) -> dict[str, Any]: ...

    def get_data_names(self) -> tuple[str, ...]: ...

    def get_split_mode(self) -> str: ...

    def get_cache_fingerprint(self) -> dict[str, Any]: ...

    def get_loader_state(self) -> LoaderState: ...

    def is_loaded(self) -> bool: ...

    def is_split_ready(self) -> bool: ...


@runtime_checkable
class CompositionDatasourceProtocol(DatasourceProtocol, Protocol):
    """Internal contract used when datasources are composed into a parent loader."""

    def get_loaded_data_for_composition(self) -> DatasetContainer: ...

    def get_split_data_for_composition(self) -> SplitDatasetContainer: ...
