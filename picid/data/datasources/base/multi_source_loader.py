"""
Concrete datasource loader for multi-source compositions.

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
from omegaconf import DictConfig, OmegaConf

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
    """
    Compose multiple child datasources into one datasource.

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
        """
        Initialize the composed loader and wire child datasources.

        Parameters
        ----------
        source_list : dict | None, optional
            Mapping from source alias to child config or loader instance.
        multisource_data_splitter : dict | None, optional
            Optional Hydra config or ``BySourceSplitter`` instance for
            source-level splits.
        **kwargs
            Passed through to ``AbstractDataSourceLoader`` and must include
            matching ``DictConfig``/dict entries for each config-driven source
            alias when not passing pre-instantiated loaders.
        """
        if source_list is None:
            source_list = {}
        kwargs.update(multisource_data_splitter=multisource_data_splitter)
        super().__init__(**kwargs)
        self._validate_constructor_inputs(
            source_list=source_list, multisource_data_splitter=multisource_data_splitter
        )

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
        self.multisource_data_splitter = self._normalize_multisource_splitter(
            multisource_data_splitter
        )

        for key in self.source_names:
            source_entry = source_list[key]
            if isinstance(source_entry, AbstractDataSourceLoader):
                datasource = source_entry
                src_params = {
                    "_kind": "instance",
                    "class": type(source_entry).__name__,
                    "data_name": source_entry.get_data_name(),
                }
                self.source_params.append(src_params)
                self._source_config_by_name[key] = deepcopy(src_params)
                self.data_source_dict[key] = datasource
                continue

            source_cfg = kwargs.get(key, source_entry)
            if source_cfg is None:
                raise DatasourceConfigurationError(
                    f"Missing source config for source '{key}'."
                )
            if isinstance(source_cfg, dict):
                source_cfg = OmegaConf.create(source_cfg)
            if not isinstance(source_cfg, DictConfig):
                raise DatasourceConfigurationError(
                    f"Source '{key}' must be an instantiated loader, dict config, or DictConfig; "
                    f"got {type(source_cfg).__name__}."
                )

            src_params = OmegaConf.to_container(source_cfg, resolve=True)
            self.source_params.append(src_params)
            self._source_config_by_name[key] = deepcopy(src_params)

            datasource = instantiate(
                source_cfg,
                is_part_of_multisource=(
                    True if self.multisource_data_splitter else False
                ),
            )
            self.data_source_dict[key] = datasource

        self._cache_config = self._build_cache_config(kwargs)

    @staticmethod
    def _normalize_multisource_splitter(multisource_data_splitter: Any):
        """
        Normalize splitter input from instance or Hydra config.

        Parameters
        ----------
        multisource_data_splitter : Any
            ``None``, an instantiated ``BySourceSplitter``, or a Hydra config
            dict/DictConfig to instantiate.

        Returns
        -------
        BySourceSplitter | None
            Ready-to-use splitter, or ``None`` when no source-level split is
            configured.
        """
        if multisource_data_splitter is None:
            return None
        if isinstance(multisource_data_splitter, BySourceSplitter):
            return multisource_data_splitter
        return instantiate(multisource_data_splitter)

    def _validate_constructor_inputs(
        self, source_list: dict[str, Any] | DictConfig, multisource_data_splitter: Any
    ) -> None:
        """
        Fail fast on invalid multisource constructor combinations.

        Parameters
        ----------
        source_list : dict[str, Any]
            Mapping from source alias to child config or loader instance.
        multisource_data_splitter : Any
            Optional splitter instance or Hydra config.
        """
        if not source_list:
            raise DatasourceConfigurationError("source_list cannot be empty.")
        if not isinstance(source_list, (dict, DictConfig)):
            raise DatasourceConfigurationError("source_list must be a dictionary.")
        for source_name in source_list.keys():
            if not isinstance(source_name, str):
                raise DatasourceConfigurationError(
                    "source_list keys must all be strings."
                )

        if multisource_data_splitter is None:
            return

        if isinstance(multisource_data_splitter, BySourceSplitter):
            return
        if isinstance(multisource_data_splitter, (dict, DictConfig)):
            return
        raise DatasourceConfigurationError(
            "multisource_data_splitter must be None, BySourceSplitter, dict, or DictConfig."
        )

    def _build_cache_config(self, raw_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Build the cache identity for the composed datasource.

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
        """
        Load every child datasource and reset parent orchestration state.

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
        """
        Materialize the combined split container for all child datasources.

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
        """
        Return the combined split dataset container.

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
        """
        Return metadata about the composed datasource.

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
        """
        Return child datasource aliases in deterministic order.

        Returns
        -------
        tuple[str, ...]
            Source aliases in the same order used for cache identity and
            multisource splitting.
        """
        return tuple(self.source_names)

    def get_multisource_data_splitter(self):
        """
        Return the instantiated multisource splitter, if any.

        Returns
        -------
        BySourceSplitter | None
            Configured source-level splitter, or ``None`` when child loaders
            own their own splits.
        """
        return self.multisource_data_splitter

    def get_cache_fingerprint(self) -> dict[str, Any]:
        """
        Return the cache identity for the composed datasource.

        Returns
        -------
        dict[str, Any]
            Explicit cache fingerprint that avoids hashing child configs twice.
        """
        return deepcopy(self._cache_config)
