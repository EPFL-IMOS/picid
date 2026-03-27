"""Loader base class for datasources that already ship with splits.

Predefined-split datasources already expose train/validation/test partitions, so
they do not participate in the normal split orchestration performed by generic
single-source or multi-source loaders.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from abc import abstractmethod
from typing import Any

from picid.data.data_objects import SplitDatasetContainer
from picid.data.datasources.base.contracts import LoaderState
from picid.data.datasources.base.exceptions import DatasourceConfigurationError
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader

logger = logging.getLogger(__name__)


class PredefinedSplitLoaderBase(AbstractDataSourceLoader):
    """Shared base for datasources that ship with predefined splits.

    Subclasses only need to provide the raw split payload; this base class
    handles the loader lifecycle and exposes the split-aware container format.
    """

    def __init__(self, **kwargs):
        """Initialize the loader and disable external split orchestration.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Datasource configuration values. ``multisource_data_splitter`` is
            rejected because predefined-split datasources already own their
            train/validation/test partitioning.
        """

        super().__init__(**kwargs)
        self._is_loaded = False
        self._is_splitted = False
        self._state = LoaderState.INITIALIZED
        self.data_dict: dict[str, Any] | None = None
        self.meta_data: dict[str, Any] = {}

        if kwargs.get("multisource_data_splitter") is not None:
            raise DatasourceConfigurationError(
                f"{self.data_name} relies on predefined splits and does not accept "
                "multisource_data_splitter."
            )
        self.multisource_data_splitter = None
        self._cache_config = self._build_cache_config()

    @abstractmethod
    def _load_data(self) -> dict[str, Any]:
        """Load the predefined split payload from the datasource backend.

        Returns
        -------
        dict[str, Any]
            Mapping of data keys to split-aware payloads.
        """

        raise NotImplementedError("Subclasses must implement _load_data().")

    def _build_cache_config(self) -> dict[str, Any]:
        """Build the cache identity for a predefined-split datasource.

        Returns
        -------
        dict[str, Any]
            Serialisable cache fingerprint that excludes ignored splitter
            compatibility kwargs.
        """
        fingerprint = {
            "data_name": self.data_name,
            "task_mode": self.task_mode,
        }
        for key, value in self._constructor_config.items():
            if key in {"data_name", "task_mode", "multisource_data_splitter"}:
                continue
            # The fingerprint mirrors meaningful constructor inputs while
            # intentionally excluding aliases already captured above.
            fingerprint[key] = deepcopy(value)
        return fingerprint

    def load_data(self) -> None:
        """Load predefined split data and mark the loader as split.

        Returns
        -------
        None
            The method updates the loader state in place.
        """
        self.data_dict = None
        self.meta_data = {}
        self._is_loaded = False
        self._is_splitted = False
        self._state = LoaderState.INITIALIZED
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_splitted = True
        self._state = LoaderState.SPLIT

    def get_data(self) -> SplitDatasetContainer:
        """Return the loaded split dataset container.

        Returns
        -------
        SplitDatasetContainer
            Split-aware dataset container ready for preprocessing.

        Raises
        ------
        DatasourceStateError
            If the predefined split payload has not been loaded yet.
        """

        self._require_loaded("getting data")
        self._require_split_ready("getting data")
        assert self.data_dict is not None, "Data container is not initialized."
        # Build the public container from a deep copy so metadata normalization
        # inside SplitDatasetContainer cannot mutate the cached loader payload.
        return SplitDatasetContainer(**deepcopy(self.data_dict))

    def get_meta_data(self) -> dict[str, Any]:
        """Return metadata associated with the predefined split payload.

        Returns
        -------
        dict[str, Any]
            Loader-specific metadata collected during ``load_data()``.

        Raises
        ------
        DatasourceStateError
            If metadata is requested before the predefined splits are loaded.
        """

        self._require_loaded("getting metadata")
        return self.meta_data

    def get_cache_fingerprint(self) -> dict[str, Any]:
        """Return a stable fingerprint for predefined-split caching.

        Returns
        -------
        dict[str, Any]
            Explicit cache fingerprint captured at construction time.
        """

        return deepcopy(self._cache_config)

    def split_data(self) -> None:
        """Log that splitting is a no-op for predefined-split datasources.

        Returns
        -------
        None
            The method intentionally leaves the already-defined splits untouched.
        """

        logger.warning(
            "%s relies on predefined splits. split_data() has no effect.",
            self.data_name,
        )
