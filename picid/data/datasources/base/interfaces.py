"""Base datasource interfaces and shared lifecycle helpers.

This module defines the public contract that preprocessing code expects from a
datasource. Concrete loaders can keep richer internal state, but they should
expose loading, splitting, naming, metadata, and cache-identity behavior
through this interface.
"""

from copy import deepcopy
import logging
from abc import ABC, abstractmethod
from typing import Any

from picid.data.data_objects import DatasetContainer, SplitDatasetContainer
from picid.data.datasources.base.contracts import LoaderState, SplitMode
from picid.data.datasources.base.exceptions import (
    DatasourceConfigurationError,
    DatasourceStateError,
)

logger = logging.getLogger(__name__)


class AbstractDataSourceLoader(ABC):
    """Common interface for all datasource loaders.

    Parameters
    ----------
    **kwargs : dict[str, Any]
        Datasource configuration resolved from Hydra or passed programmatically.
        At minimum, ``data_name`` and ``task_mode`` must be present.

    Attributes
    ----------
    data_name : str
        Stable datasource identifier exposed to downstream consumers.
    task_mode : str
        Task type associated with the datasource, for example ``"rul"`` or
        ``"fault"``.
    multisource_data_splitter : Any | None
        Optional multisource splitter configuration or instantiated splitter.
    debug_subset_range : Any | None
        Optional debug slicing configuration preserved for concrete loaders.
    debug_subsample_rate : Any | None
        Optional debug subsampling factor preserved for concrete loaders.

    Notes
    -----
    ``_repr_config`` is retained only as a debug snapshot for ``__repr__``
    output. Contract behavior should prefer explicit attributes and the frozen
    constructor snapshot exposed through ``get_cache_fingerprint()``.
    """

    def __init__(self, **kwargs):
        """Initialize shared datasource state.

        Parameters
        ----------
        **kwargs : dict[str, Any]
            Datasource configuration values.
        """

        self.data_name = kwargs["data_name"]
        self.task_mode = kwargs["task_mode"]
        self.multisource_data_splitter = kwargs.get("multisource_data_splitter")
        self._state = LoaderState.INITIALIZED
        self._repr_config: dict[str, Any] = dict(kwargs)
        # Keep a frozen view of constructor inputs so cache identity can be
        # derived without reading back mutable compatibility state later on.
        self._constructor_config: dict[str, Any] = deepcopy(kwargs)
        self._cache_config: dict[str, Any] = deepcopy(self._constructor_config)

        # Debug options (still supported, but not the only way to cache)
        self.debug_subset_range = kwargs.get("debug_subset_range", None)
        self.debug_subsample_rate = kwargs.get("debug_subsample_rate", None)

    def get_split_mode(self) -> str:
        """Return the canonical split mode for the datasource.

        Returns
        -------
        str
            ``"within_units"`` when each source owns its own train/val/test
            split, otherwise ``"between_units"`` when a multisource splitter is
            responsible for partitioning sources.
        """

        if self.multisource_data_splitter is None:
            return SplitMode.WITHIN_UNITS.value
        return SplitMode.BETWEEN_UNITS.value

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._repr_config.items())
        return f"{self.__class__.__name__}({args})"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load_data(self) -> None:
        """Load raw datasource content into the loader state."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def split_data(self) -> None:
        """Create or materialize datasource splits.

        Concrete loaders may either delegate splitting to child loaders,
        perform dataset-specific partitioning, or no-op when splits are already
        predefined by the dataset.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_data(self) -> DatasetContainer | SplitDatasetContainer:
        """Return the current loaded dataset container."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_meta_data(self) -> dict[str, Any]:
        """Return metadata associated with the loaded datasource.

        Returns
        -------
        dict[str, Any]
            Loader-specific metadata describing units, provenance, or split
            context.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def get_data_name(self) -> str | list[str]:
        """Return a compatibility name view for legacy callers.

        Returns
        -------
        str | list[str]
            A single string for single-datasource loaders, or a list of names
            for loaders that expose multiple logical datasource names.
        """
        data_names = self.get_data_names()
        if len(data_names) == 1:
            return data_names[0]
        return list(data_names)

    def get_data_names(self) -> tuple[str, ...]:
        """Return canonical datasource names as a tuple.

        Returns
        -------
        tuple[str, ...]
            Stable datasource identifiers. Concrete multi-source loaders should
            return the outer datasource identity here and expose inner source
            names separately if needed.
        """
        return (self.data_name,)

    # ------------------------------------------------------------------
    # Metadata and cache identity
    # ------------------------------------------------------------------

    def get_cache_fingerprint(self) -> dict[str, Any]:
        """Return a stable config snapshot for datasource-level cache keys.

        Returns
        -------
        dict[str, Any]
            Serialisable configuration that should invalidate cached pipeline
            stages when datasource behavior changes.
        """
        return deepcopy(self._cache_config)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def get_loader_state(self) -> LoaderState:
        """Return the current lifecycle state for the datasource.

        Returns
        -------
        LoaderState
            Current lifecycle marker tracked by the shared datasource base.
        """

        return self._state

    def is_loaded(self) -> bool:
        """Return whether the datasource has materialized data.

        Returns
        -------
        bool
            ``True`` once the loader has progressed beyond initialization.
        """

        return self._state in {LoaderState.LOADED, LoaderState.SPLIT} or bool(
            getattr(self, "_is_loaded", False)
        )

    def is_split_ready(self) -> bool:
        """Return whether the datasource exposes split-aware data.

        Returns
        -------
        bool
            ``True`` when the datasource lifecycle has reached the split stage.
        """

        return self._state == LoaderState.SPLIT or bool(
            getattr(self, "_is_splitted", False)
        )

    # ------------------------------------------------------------------
    # Guard helpers
    # ------------------------------------------------------------------

    def _require_loaded(self, action: str) -> None:
        """Ensure the datasource has completed its load phase.

        Parameters
        ----------
        action : str
            Human-readable description of the attempted operation.

        Returns
        -------
        None
            The method returns silently when the datasource is loaded.

        Raises
        ------
        DatasourceStateError
            If the datasource has not been loaded yet.
        """

        if not self.is_loaded():
            raise DatasourceStateError(f"Data must be loaded before {action}.")

    def _require_split_ready(self, action: str) -> None:
        """Ensure the datasource has materialized split-aware data.

        Parameters
        ----------
        action : str
            Human-readable description of the attempted operation.

        Returns
        -------
        None
            The method returns silently when split-aware data is available.

        Raises
        ------
        DatasourceStateError
            If the datasource has not reached the split lifecycle state.
        """

        if not self.is_split_ready():
            raise DatasourceStateError(f"Data must be splitted before {action}.")

    def _require_configuration(self, condition: bool, message: str) -> None:
        """Validate a caller-visible datasource configuration precondition.

        Parameters
        ----------
        condition : bool
            Whether the configuration precondition is satisfied.
        message : str
            Error message to surface when the configuration is invalid.

        Returns
        -------
        None
            The method returns silently when the condition is satisfied.

        Raises
        ------
        DatasourceConfigurationError
            If the configuration precondition fails.
        """

        if not condition:
            raise DatasourceConfigurationError(message)

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def get_loaded_data_for_composition(self) -> DatasetContainer:
        """Return an unsplit container for parent datasource composition.

        Returns
        -------
        DatasetContainer
            Unsplit dataset payload.

        Raises
        ------
        TypeError
            If the datasource only exposes split-aware data through its public
            contract.
        """

        data = self.get_data()
        if isinstance(data, SplitDatasetContainer):
            raise TypeError(
                f"{self.__class__.__name__} does not expose an unsplit DatasetContainer for multisource composition."
            )
        if not isinstance(data, DatasetContainer):
            raise TypeError(
                f"{self.__class__.__name__} returned an unexpected data container: {type(data).__name__}"
            )
        return data

    def get_split_data_for_composition(self) -> SplitDatasetContainer:
        """Return a split-aware container for parent datasource composition.

        Returns
        -------
        SplitDatasetContainer
            Split dataset payload ready for split-wise composition.

        Raises
        ------
        TypeError
            If the datasource does not expose split-aware data through its
            public contract.
        """

        data = self.get_data()
        if not isinstance(data, SplitDatasetContainer):
            raise TypeError(
                f"{self.__class__.__name__} does not expose a SplitDatasetContainer for multisource composition."
            )
        return data
