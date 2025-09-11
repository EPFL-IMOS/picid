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

        # Cache options
        self.use_cache = kwargs.get("use_cache", False)
        self.cache_path = kwargs.get("cache_path", None)
        self.config_hash = hash_config(kwargs)

        # Debug options (still supported, but not the only way to cache)
        self.debug_subset_range = kwargs.get("debug_subset_range", None)
        self.debug_subsample_rate = kwargs.get("debug_subsample_rate", None)

        # If caching enabled, validate and prepare directory
        if self.use_cache:
            if self.cache_path is None:
                raise ValueError("use_cache is True, but cache_path is not set.")

            cache_path = Path(self.cache_path).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache_path = cache_path
    
    def _get_cache_paths(self) -> dict:
        """Return paths for dataset, configuration JSON, and hash file."""
        return {
            "folder": self.cache_path,
            "data": self.cache_path / "data.pkl",
            "config": self.cache_path / "config.json",
            "hash": self.cache_path / "hash.txt",
        }

    def _is_cache_valid(self, paths: dict) -> bool:
        """Check if cache exists and hash matches."""
        required = ["data", "config", "hash"]
        if not all(paths[k].exists() for k in required):
            return False

        try:
            with open(paths["hash"], "r") as f:
                stored_hash = f.read().strip()
        except Exception:
            return False

        return stored_hash == self.config_hash

    def _load_from_cache(self):
        """Load dataset from cache if valid, else return None."""
        paths = self._get_cache_paths()
        if self._is_cache_valid(paths):
            with open(paths["data"], "rb") as f:
                data = pickle.load(f)
            
            # with open(paths["config"], "r") as f:
            #     self._init_kwargs.update(json.load(f))

            logger.info(f"Loaded {self.data_name} from cache at {paths['folder']}")
            return data
        return None

    def _save_to_cache(self, data):
        """Save dataset, config, and hash to cache."""
        paths = self._get_cache_paths()

        # Save data
        with open(paths["data"], "wb") as f:
            pickle.dump(data, f)

        # Save config
        with open(paths["config"], "w") as f:
            json.dump(ensure_serializable(self._init_kwargs), f, indent=2)

        # Save hash
        with open(paths["hash"], "w") as f:
            f.write(self.config_hash)

        logger.info(f"Saved {self.data_name} cache at {paths['folder']}")

    def handle_cache(self, build_data_fn):
        """
        Full cache handling pipeline:
        - Try load from cache.
        - If invalid or missing → build data via `build_data_fn` and save cache.
        """
        if not self.use_cache:
            return build_data_fn()

        # Attempt to load
        cached_data = self._load_from_cache()
        if cached_data is not None:
            return cached_data

        # Build new data and save
        data = build_data_fn()
        self._save_to_cache(data)
        return data
    
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