from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
import os
import numpy as np

from lmetk.data.data_objects.data import DatasetContainer, SplitDatasetContainer
from lmetk.utils.hash_utils import hash_config, ensure_serializable
import json

import pickle
import logging
logger = logging.getLogger(__name__)
class AbstractDataSourceLoader(ABC):
    """
    Abstract base class for data source loaders in the TabPHM pipeline.
    This interface defines the required methods and initialization logic for loading,
    splitting, and managing datasets from various sources. It supports configuration
    options for debugging, caching, and data splitting modes.

    Notes:
        - Raises ValueError if debug_cache_dataset is True but debug_cache_path is not set.
    """

    def __init__(self, **kwargs):
        """
        Initializes the object with configuration parameters.

        Args:
            data_name (str): Name of the dataset.
            task_mode (str): Mode of the task (e.g., classification, regression).
            debug_cache_dataset (bool, optional): If True, enables caching
                of the dataset for debugging. Defaults to False.
            debug_cache_path (str, optional): Path to store the cached dataset.
                Required if debug_cache_dataset is True.
            debug_subset_range (Any, optional): Range of subset for debugging purposes.
                Defaults to None.
            debug_subsample_rate (Any, optional): Subsample rate for debugging purposes.
                Defaults to None.
        Raises:
            ValueError: If debug_cache_dataset is True but debug_cache_path is not set.
        """

        self._init_kwargs = kwargs
        self.data_name = kwargs["data_name"]
        self.task_mode = kwargs["task_mode"]

        # Debug options (still supported, but not the only way to cache)
        self.debug_subset_range = kwargs.get("debug_subset_range", None)
        self.debug_subsample_rate = kwargs.get("debug_subsample_rate", None)
    
    def get_split_mode(self):
        """
        Determines the data split mode based on the presence of a multisource data splitter.
        Returns:
            str: The split mode, either "per_unit" if no multisource data splitter is provided,
            or "cross_unit" if a multisource data splitter is present.
        """

        multisource_data_splitter = self._init_kwargs.get(
            "multisource_data_splitter", None
        )

        if not multisource_data_splitter:
            split_mode = "per_unit"
        else:
            split_mode = "cross_unit"

        return split_mode

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}({args})"

    @abstractmethod
    def load_data(self) -> None:
        """Load the data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_data(self) -> DatasetContainer | SplitDatasetContainer:
        """Get the data after it is loaded."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_data_name(self) -> str:
        """Return source data names."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def split_data(self) -> None:
        """
        Splits the provided data list into separate sources using the configured
        multisource data splitter.
        Args:
            data_lst (list[Dict]): A list of data dictionaries to be split by source.
            source_names (list[str]): A list of source names corresponding to the data.
        Returns:
            Tuple[Dict, list[str]]:
                - data_dict: A dictionary mapping source names to their respective split data.
                - source_names_tree_view: A list of source names structured
                    for tree view representation.
        Raises:
            AssertionError: If `self.multisource_data_splitter` is not an
                instance of `BySourceSplitter`.
        """
        raise NotImplementedError("Subclasses must implement this method.")