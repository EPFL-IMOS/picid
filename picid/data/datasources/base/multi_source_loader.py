"""Concrete datasource loader for multi-source compositions.

The multi-source loader orchestrates a set of child loaders and exposes the
combined dataset as a single datasource to preprocessing. It supports two
composition modes:

1. each child datasource performs its own split and the parent concatenates the
   per-split units; or
2. a multisource splitter partitions whole child payloads across train/val/test.
"""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, override

from hydra.utils import instantiate
from omegaconf import OmegaConf

from picid.data.data_objects import DatasetContainer, SplitDatasetContainer
from picid.data.datasources.base.exceptions import (
    DatasourceCompositionError,
    DatasourceConfigurationError,
    DatasourceStateError,
)
from picid.data.datasources.base.contracts import (
    CompositionDatasourceProtocol,
    LoaderState,
)
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.data.split_strategies import BySourceSplitter

logger = logging.getLogger(__name__)


class MultiSourceLoader(AbstractDataSourceLoader):
    """Compose multiple child datasources into one datasource.

    Parameters
    ----------
    source_list : dict | None, optional
        Mapping from source alias to child datasource config. The alias is used
        for source-level parity and multisource split decisions.
    multisource_data_splitter : dict | None, optional
        Hydra config for a splitter that assigns entire sources to train/val/test.
        When omitted, each child loader is expected to split itself.
    **kwargs : dict[str, Any]
        Additional datasource configuration, including one config entry for each
        source alias in ``source_list``. Each entry may be a plain ``dict`` or an
        OmegaConf ``DictConfig``; plain dicts are wrapped before
        ``OmegaConf.to_container`` and Hydra ``instantiate``.

    Notes
    -----
    ``get_data_names()`` returns the outer datasource identity. Use
    ``get_source_names()`` to inspect the child-source aliases.
    """

    def __init__(
        self,
        source_list: Optional[dict] = None,
        multisource_data_splitter: Optional[dict] = None,
        **kwargs,
    ):
        if source_list is None:
            source_list = {}
        kwargs.update(multisource_data_splitter=multisource_data_splitter)
        super().__init__(**kwargs)

        self.data_source_dict: dict[str, CompositionDatasourceProtocol] = {}
        self.source_params: list[dict] = []
        self.source_names: list[str] = list(source_list.keys())
        self._source_config_by_name: dict[str, dict[str, Any]] = {}

        self.data_lst: list[DatasetContainer] = []
        self.container: Optional[DatasetContainer] = None
        self._multisource_splitter_config = deepcopy(multisource_data_splitter)

        self._is_splitted = False
        self._is_loaded = False

        # Instantiate multisource_data_splitter if is not none
        self.multisource_data_splitter = (
            instantiate(multisource_data_splitter)
            if multisource_data_splitter is not None
            else None
        )

        for key in self.source_names:
            src_params = OmegaConf.to_container(kwargs[key], resolve=True)
            self.source_params.append(src_params)
            self._source_config_by_name[key] = deepcopy(src_params)

            datasource = instantiate(
                kwargs[key],
                is_part_of_multisource=(
                    True if self.multisource_data_splitter else False
                ),
            )
            self.data_source_dict[key] = datasource

        self._cache_config = self._build_cache_config(kwargs)

    def _build_cache_config(self, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the cache identity for the composed datasource.

        Parameters
        ----------
        raw_kwargs : dict[str, Any]
            Original constructor kwargs before child datasources were
            instantiated.

        Returns
        -------
        dict[str, Any]
            Serialisable cache identity that captures the outer datasource,
            the multisource splitter config, child config order, and any extra
            top-level knobs without hashing child configs twice.
        """
        ignored_keys = {
            "data_name",
            "task_mode",
            "multisource_data_splitter",
            "source_list",
        }
        ignored_keys.update(self.source_names)

        config = {
            "data_name": self.data_name,
            "task_mode": self.task_mode,
            "multisource_data_splitter": deepcopy(self._multisource_splitter_config),
            "source_order": list(self.source_names),
            "source_list": deepcopy(self._source_config_by_name),
        }
        for key, value in raw_kwargs.items():
            if key in ignored_keys:
                continue
            config[key] = deepcopy(value)
        return config

    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    @override
    def load_data(self):
        """Load every child datasource and reset parent orchestration state.

        Returns
        -------
        None
            The method updates loader state in place.
        """
        self.data_lst = []
        self.container = None
        self._is_loaded = False
        self._is_splitted = False
        self._state = LoaderState.INITIALIZED

        for ds in self.data_source_dict.values():
            ds.load_data()
            if not ds.is_loaded():
                logger.warning(f"Data source '{ds}' failed to load.")

        if all(ds.is_loaded() for ds in self.data_source_dict.values()):
            self._is_loaded = True
            self._state = LoaderState.LOADED

    def split_data(self):
        """Materialize the combined split container for all child datasources.

        Raises
        ------
        DatasourceStateError
            If a child datasource was not loaded or violates the parent split
            strategy.
        DatasourceCompositionError
            If multisource composition encounters incompatible child container
            types.
        DatasourceConfigurationError
            If the configured multisource splitter is not supported.
        """

        data_lst = []
        source_names = self.data_source_dict.keys()

        self._require_loaded("splitting")

        if self._is_splitted:
            logger.warning(
                "Attempting to load data twice. Data has already been loaded."
            )
            return

        for key in source_names:
            datasource = self.data_source_dict[key]
            if not datasource.is_loaded():
                raise DatasourceStateError(f"Source {key} has not been loaded")

            # Validate splitting logic based on presence of multisource_data_splitter
            if self.multisource_data_splitter:
                if datasource.is_split_ready():
                    raise DatasourceStateError(
                        f"multisource_data_splitter is provided, but source "
                        f"'{key}' is already split. This configuration is not supported."
                    )
            else:
                # When no multisource splitter: each child splits itself before get_data().
                if not datasource.is_split_ready():
                    datasource.split_data()

            if self.multisource_data_splitter:
                data_lst.append(datasource.get_loaded_data_for_composition())
            else:
                data_lst.append(datasource.get_split_data_for_composition())

        if self.multisource_data_splitter:
            instance_clss = {
                v for chunk in data_lst for v in chunk.get_instance_cls().values()
            }
            if len(instance_clss) != 1:
                raise DatasourceCompositionError(
                    f"All data sources must have the same instance_cls, got {instance_clss}"
                )

            if not isinstance(self.multisource_data_splitter, BySourceSplitter):
                raise DatasourceConfigurationError(
                    "multisource_data_splitter must be an instance of BySourceSplitter"
                )
            # In by-source mode we split at the source boundary, so each child
            # contributes one unsplit payload to the multisource splitter.
            data_dict, _ = self.multisource_data_splitter(data_lst, source_names)
            self.container = SplitDatasetContainer(**data_dict)
        else:
            instance_clss = {
                v for chunk in data_lst for v in chunk.get_instance_cls().values()
            }

            data_names = list(data_lst[0].keys())
            inner_split = list(data_lst[0][data_names[0]].keys())
            data_dict = {
                data_name: {split_name: [] for split_name in inner_split}
                for data_name in data_names
            }

            # Without a multisource splitter we concatenate the already-split
            # child units into one split container per data key.
            for data in data_lst:
                for split_name in inner_split:
                    for data_name in data_names:
                        data_dict[data_name][split_name].extend(
                            data[data_name][split_name]
                        )

            self.container = SplitDatasetContainer(**data_dict)

        self._is_loaded = True
        self._is_splitted = True
        self._state = LoaderState.SPLIT

    @override
    def get_data(self) -> SplitDatasetContainer:
        """Return the combined split dataset container.

        Returns
        -------
        SplitDatasetContainer
            Fully materialized parent split container.

        Raises
        ------
        DatasourceStateError
            If the composed datasource has not been loaded and split yet.
        """
        self._require_loaded("getting data")
        self._require_split_ready("getting data")
        assert self.container is not None, "Data container is not initialized."
        return self.container

    def get_meta_data(self) -> Dict[str, Any]:
        """Return metadata about the composed datasource.

        Returns
        -------
        dict[str, Any]
            Currently empty metadata placeholder for the composed loader.

        Raises
        ------
        DatasourceStateError
            If metadata is requested before the parent datasource is loaded.
        """
        self._require_loaded("getting metadata")
        return {}

    def get_source_names(self) -> tuple[str, ...]:
        """Return child datasource aliases in deterministic order.

        Returns
        -------
        tuple[str, ...]
            Source aliases in the same order used for cache identity and
            multisource splitting.
        """
        return tuple(self.source_names)

    def get_multisource_data_splitter(self):
        """Return the instantiated multisource splitter, if any.

        Returns
        -------
        BySourceSplitter | None
            Configured source-level splitter, or ``None`` when child loaders
            own their own splits.
        """
        return self.multisource_data_splitter

    def get_cache_fingerprint(self) -> dict[str, Any]:
        """Return the cache identity for the composed datasource.

        Returns
        -------
        dict[str, Any]
            Explicit cache fingerprint that avoids hashing child configs twice.
        """
        return deepcopy(self._cache_config)
