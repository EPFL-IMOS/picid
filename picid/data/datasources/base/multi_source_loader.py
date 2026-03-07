import logging
from typing import Any, Dict, Optional, override

from hydra.utils import instantiate
from omegaconf import OmegaConf

from picid.data.data_objects.data import DatasetContainer, SplitDatasetContainer
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.data.preprocessing.splits import BySourceSplitter

logger = logging.getLogger(__name__)


class MultiSourceLoader(AbstractDataSourceLoader):
    """
    Loader for multiple data sources, supporting both individual and multisource data splitting.
    This class manages the instantiation and loading of multiple data sources, optionally applying
    a multisource data splitter for joint data splitting across sources. It provides methods to
    retrieve loaded data, metadata, source names, and the splitter instance.

    Parameters
    ----------
    **kwargs
        Keyword arguments containing configuration for each data source and optional
        multisource_data_splitter. Expected keys:
            - source_list: dict mapping source names to their configuration.
            - multisource_data_splitter: Optional splitter instance for joint splitting.
            - <source_name>: Configuration for each individual source.

    Attributes
    ----------
    data_source_dict : dict
        Mapping of source names to instantiated data source objects.
    source_params : list
        List of parameter dicts for each source.
    source_names : list
        List of source names.
    multisource_data_splitter : Optional[BySourceSplitter]
        Instance for multisource splitting.

    Methods
    -------
    __repr__
        Returns a string representation of the loader.
    get_data
        Loads and returns data from all sources, applying splitting logic as configured.
    get_meta_data
        Returns metadata about the loaded data sources.
    get_data_name
        Returns the list of source data names.
    get_multisource_data_splitter
        Returns the multisource data splitter instance.
    split_data
        Applies the multisource data splitter to the data.
    """

    def __init__(
        self,
        source_list: Optional[dict] = {},
        multisource_data_splitter: Optional[dict] = None,
        **kwargs,
    ):
        kwargs.update(multisource_data_splitter=multisource_data_splitter)
        super().__init__(**kwargs)

        self.data_source_dict: dict[SingleSourceLoader] = {}
        self.source_params: list[dict] = []
        self.source_names: list[str] = list(source_list.keys())

        self.data_lst: list[DatasetContainer] = []
        self.container: Optional[DatasetContainer] = None

        self._is_splitted = False
        self._is_loaded = False

        # Instantiate multisource_data_splitter if is not none
        self.multisource_data_splitter = multisource_data_splitter

        if not isinstance(self.multisource_data_splitter, BySourceSplitter):
            self.multisource_data_splitter = (
                instantiate(multisource_data_splitter)
                if multisource_data_splitter is not None
                else None
            )

        for key, value in source_list.items():
            if isinstance(value, dict):
                src_params = OmegaConf.to_container(kwargs[key], resolve=True)
                self.source_params.append(src_params)

                datasource = instantiate(
                    kwargs[key],
                    is_part_of_multisource=(
                        True if self.multisource_data_splitter else False
                    ),
                )
            else:
                datasource = value

            self.data_source_dict[key] = datasource

    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    @override
    def load_data(self):
        """
        Loads data from all data sources in the data_source_dict.
        Iterates over each data source object in the data_source_dict and calls its
        load_data() method to perform the data loading operation.

        Returns
        -------
        None
        """
        for ds in self.data_source_dict.values():
            ds.load_data()
            if not ds._is_loaded:
                logger.warning(f"Data source '{ds}' failed to load.")

        if all(
            getattr(ds, "_is_loaded", False) for ds in self.data_source_dict.values()
        ):
            self._is_loaded = True

    def split_data(self):
        """
        Splits and loads data from multiple sources according to the configuration.
        This method iterates over all data sources in `self.data_source_dict`,
        ensuring each source is loaded and appropriately split.
        If a `multisource_data_splitter` is provided, it validates that none of the sources
        are already split and uses the splitter to combine data from all sources.
        If no splitter is provided, each source is split individually.

        Raises
        ------
        RuntimeError
            If attempting to load data twice, if a source is not loaded,
            or if the splitting configuration is invalid.

        Side Effects
        ------------
            - Populates `self.container` with the split data.
            - Sets `self._is_loaded` to True after successful loading and splitting.
        """

        data_lst = []
        source_names = self.data_source_dict.keys()

        assert self._is_loaded, "Data must be loaded before splitting."

        if self._is_splitted:
            logger.warning(
                "Attempting to load data twice. Data has already been loaded."
            )
            return

        for key in source_names:
            datasource = self.data_source_dict[key]
            if not getattr(datasource, "_is_loaded", False):
                raise RuntimeError(f"Source {key} has not been loaded")

            # Validate splitting logic based on presence of multisource_data_splitter
            if self.multisource_data_splitter:
                if getattr(datasource, "_is_splitted", False):
                    raise RuntimeError(
                        f"multisource_data_splitter is provided, but source "
                        f"'{key}' is already split. This configuration is not supported."
                    )
            else:
                # When no multisource splitter: each child splits itself before get_data().
                if not getattr(datasource, "_is_splitted", False):
                    datasource.split_data()

            data_lst.append(datasource.get_data())

        if self.multisource_data_splitter:
            instance_clss = {
                v for chunk in data_lst for v in chunk.get_instance_cls().values()
            }
            if len(instance_clss) != 1:
                raise TypeError(
                    f"All data sources must have the same instance_cls, got {instance_clss}"
                )

            assert isinstance(
                self.multisource_data_splitter, BySourceSplitter
            ), "multisource_data_splitter must be an instance of BySourceSplitter"
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

            for data in data_lst:
                for split_name in inner_split:
                    for data_name in data_names:
                        data_dict[data_name][split_name].extend(
                            data[data_name][split_name]
                        )

            self.container = SplitDatasetContainer(**data_dict)

        self._is_loaded = True

    @override
    def get_data(self) -> SplitDatasetContainer:
        """Return data sources.

        Returns
        -------
        dict | list[dict]
            Either dict in case we use multisource_data_splitter
            or list of dicts in case we use data_splitter

        """
        if self._is_loaded is False:
            assert (
                self.container is not None
            ), "Found initialized data container, but source was never loaded."
            raise AssertionError("Data source has not been loaded.")
        else:
            assert self.container is not None, "Data container is not initialized."
            return self.container

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return {}

    @override
    def get_data_name(self) -> list[str]:
        return self.source_names

    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter
