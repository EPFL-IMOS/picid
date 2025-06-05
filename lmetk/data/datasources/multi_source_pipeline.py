from typing import Dict, List

import numpy as np
from box import Box
from omegaconf import DictConfig
from sklearn.base import TransformerMixin

from lmetk.data.datasources.base import TimeSeriesDatasetInterface


class MultiSourcePipeline(TimeSeriesDatasetInterface):

    def __init__(self, load_arguments: DictConfig, 
                 data_splitter, 
                 scalers: Dict[str, TransformerMixin], 
                 source_list,
                 **kwargs):
        
        super().__init__(load_arguments, data_splitter, scalers)

        self.load_arguments: DictConfig = load_arguments
        self.data_splitter = data_splitter
        self.scalers: Dict[str, TransformerMixin] = scalers
        self.data_dict: (
            Dict[str, Dict[str, np.ndarray]] | Box
        ) = {}  # centralized storage

        self.sources: List[TimeSeriesDatasetInterface] = source_list


    def get_data(self):
        """Return splitted data and timestamps."""


    def get_scaler(self):
        return self.scalers


    def _pipeline(self):
        self._load_data()
        self._generate_splits()
        self._apply_transformations()

        for src in self.sources:
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
        for src in self.sources:
            # Ensure each source has its data loaded but not yet split or scaled
            src._load_data()


    def _generate_splits(self):
        """Split the data and timestamps into train/val/test sets."""
        for src in self.sources:
            src._generate_splits()


    def _apply_transformations(self):
        """Apply any preprocessing, such as scaling."""
        for src in self.sources:
            src._apply_transformations()


    def __repr__(self) -> str:
        return (
            f"MultiSourceDataSource(sources={len(self.sources)} units, "
            f"load_arguments={self.load_arguments}, "
            f"data_splitter={self.data_splitter}, scaler={self.scaler})"
        )
