from typing import Dict, List

import numpy as np
from box import Box
from omegaconf import DictConfig
from sklearn.base import TransformerMixin

from lmetk.data.datasources.base import TimeSeriesDatasetInterface
from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter


class MultiSourcePipeline(TimeSeriesDatasetInterface):

    def __init__(self, load_arguments: DictConfig, 
                 data_splitter, 
                 scalers: Dict[str, TransformerMixin], 
                 source_list,
                 source_names: Optional[List[str]] = None,
                 **kwargs):
        
        super().__init__(load_arguments, data_splitter, scalers)

        self.load_arguments: DictConfig = load_arguments
        self.data_splitter = data_splitter

        if data_splitter is None:
            assert scalers is None, \
                "If data_splitter is None, scalers must also be None. " \

        self._is_loaded = False  # Flag to check if data is loaded
        self._is_split = False  # Flag to check if data is split    
        self._is_transformed = False  # Flag to check if data is transformed

        assert isinstance(data_splitter, BySourceSplitter) or data_splitter is None, \
            "data_splitter must be an instance of BySourceSplitter or None"

        self.scalers: Dict[str, TransformerMixin] = scalers
        self.data_dict: (
            Dict[str, Dict[str, np.ndarray]] | Box
        ) = {}  # centralized storage

        self.sources: Dict[str, TimeSeriesDatasetInterface] = source_list
        self.split_sources: Dict[str, Dict [str, TimeSeriesDatasetInterface]] = {}


    def get_data(self):
        """Return splitted data and timestamps."""


    def get_scaler(self):
        return self.scalers


    def _pipeline(self):
        self._load_data()
        self._generate_splits()
        self._apply_transformations()

        for source_key, src in self.sources.items():
            for key, value in src.data_dict.items(): # features, timestamps etc.
                if key not in self.data_dict:
                    self.data_dict[key] = {}

                # collect all the data from each source
                for sub_key, sub_value in value.items(): # train, val, test
                    if sub_key not in self.data_dict[key]:
                        self.data_dict[key][sub_key] = []
                    self.data_dict[key][sub_key].append(sub_value)

        self.data_dict = Box(
            self.data_dict
        )  # Convert to a Box object for easier access


    def _load_data(self):
        """
        Load and concatenate data and timestamps from all sources.
        Returns combined data, combined timestamps, and an optional mask.
        """
        for key, src in self.sources.items():
            # Ensure each source has its data loaded but not yet split or scaled
            assert not src._is_loaded, (
                f"Source {key} has already been loaded. "
                "If you want to re-load, please reset the data source."
            )
            src._load_data()

        self._is_loaded = True

    def _generate_splits(self):
        """Split the data and timestamps into train/val/test sets."""

        assert self._is_loaded, "Data must be loaded before splitting."
        assert not self._is_split, (
            "Data has already been split. "
            "If you want to re-split, please reset the data source."
        )

        if self.data_splitter is None:
            # If no data_splitter is provided, generate splits for each source
            for key, src in self.sources.items():
                assert not src._is_split, (
                    f"Source {key} has already been split. "
                    "If you want to re-split, please reset the data source."
                )
                src._generate_splits()
        else:
            # Split on the source level
            self.split_sources = self.data_splitter.split_data(self.sources)

        self._is_split = True  # Mark data as split


    def _apply_transformations(self):

        assert self._is_split, "Data must be split before applying transformations."
        assert self._is_loaded, "Data must be loaded before applying transformations."
        assert not self._is_transformed, (
            "Data has already been transformed. "
            "If you want to re-transform, please reset the data source."
        )
        # assert if split_sources is not empty dict
        assert self.split_sources, "Split sources must not be empty."


        if self.scalers is None:
            """Apply any preprocessing, such as scaling."""
            for key, src in self.sources.items():
                assert not src._is_transformed, (
                    f"Source {key} has already been transformed. "
                    "If you want to re-transform, please reset the data source."
                )
                src._apply_transformations()
        else:
            if self.load_arguments.get("scale", True):

                # fit
                for key, scaler in self.scalers.items():
                    scaler.fit_sources(self.split_sources["train"], key)

                # transform
                for split_name in ["train", "val", "test"]:
                    if split_name in self.data_dict["features"]:
                        for key, scaler in self.scalers.items():
                            self.data_dict[key][split_name] = scaler.transform(
                                self.data_dict[key][split_name]
                            )

        self._is_transformed = True  # Mark data as transformed

    def __repr__(self) -> str:
        return (
            f"MultiSourceDataSource(sources={len(self.sources)} units, "
            f"load_arguments={self.load_arguments}, "
            f"data_splitter={self.data_splitter}, scaler={self.scaler})"
        )
