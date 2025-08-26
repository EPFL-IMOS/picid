from typing import Any, Dict, Optional, override

from hydra.utils import instantiate
from omegaconf import OmegaConf

from lmetk.data.datasources.base.interfaces import DataSourceLoaderInterface
from lmetk.data.datasources.base.single_source_loader import SingleSourceLoader
from lmetk.data.preprocessing.splits import BySourceSplitter


class MultiSourceLoader(DataSourceLoaderInterface):
    """
    Loader for multiple data sources, supporting both individual and multisource data splitting.
    This class manages the instantiation and loading of multiple data sources, optionally applying
    a multisource data splitter for joint data splitting across sources. It provides methods to
    retrieve loaded data, metadata, source names, and the splitter instance.
    Args:
        **kwargs: Keyword arguments containing configuration for each data source and optional
            multisource_data_splitter. Expected keys:
                - source_list: dict mapping source names to their configuration.
                - multisource_data_splitter: Optional splitter instance for joint splitting.
                - <source_name>: Configuration for each individual source.
    Attributes:
        data_source_dict (dict): Mapping of source names to instantiated data source objects.
        source_params (list): List of parameter dicts for each source.
        source_names (list): List of source names.
        multisource_data_splitter (Optional[BySourceSplitter]): Instance for multisource splitting.
    Methods:
        __repr__(): Returns a string representation of the loader.
        get_data(): Loads and returns data from all sources, applying splitting logic as configured.
        get_meta_data(): Returns metadata about the loaded data sources.
        get_data_name(): Returns the list of source data names.
        get_multisource_data_splitter(): Returns the multisource data splitter instance.
        split_data(data_lst, source_names): Applies the multisource data splitter to the data.
    """

    def __init__(
        self,
        source_list: Optional[dict] = {},
        multisource_data_splitter: Optional[dict] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.data_source_dict: dict[SingleSourceLoader] = {}
        self.source_params: list[dict] = []
        self.source_names: list[str] = list(source_list.keys())

        # Instantiate multisource_data_splitter if is not none
        self.multisource_data_splitter = (
            instantiate(multisource_data_splitter)
            if multisource_data_splitter is not None
            else None
        )

        for key in self.source_names:
            src_params = OmegaConf.to_container(kwargs[key], resolve=True)
            self.source_params.append(src_params)

            datasource = instantiate(
                kwargs[key],
                multisource_data_splitter=(
                    True if self.multisource_data_splitter else False
                ),
            )
            self.data_source_dict[key] = datasource

    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    @override
    def load_data(self):
        """
        Loads data from all data sources in the data_source_dict.
        Iterates over each data source object in the data_source_dict and calls its
        load_data() method to perform the data loading operation.
        Returns:
            None
        """
        for ds in self.data_source_dict.values():
            ds.load_data()

    @override
    def get_data(self) -> dict | list[dict]:
        """Return data sources.
        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter
            or list of dicts in case we use data_splitter

        """
        data_lst = []
        source_names = self.data_source_dict.keys()
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
                if not getattr(datasource, "_is_splitted", False):
                    raise RuntimeError(
                        f"Source '{key}' must be split when multisource_data_splitter is not provided."
                    )

            data_lst.append(datasource.get_data())

        if self.multisource_data_splitter:
            data_dict, _ = self.split_data(data_lst, source_names)
            self._is_loaded = True
            return data_dict

        else:
            return data_lst

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

    @override
    def split_data(self, data_lst: list[Dict], source_names: list[str]):
        assert isinstance(
            self.multisource_data_splitter, BySourceSplitter
        ), "multisource_data_splitter must be an instance of BySourceSplitter"
        # TODO: Normalize input output, why
        data_dict, source_names_tree_view = self.multisource_data_splitter(
            data_lst, source_names
        )
        return data_dict, source_names_tree_view
