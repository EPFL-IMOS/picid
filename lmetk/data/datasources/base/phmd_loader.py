import logging
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, override

import numpy as np
import pandas as pd
from lmetk.data.data_objects.data import DatasetContainer, SplitDatasetContainer
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader
from phmd import datasets

logger = logging.getLogger(__name__)


def convert_outer_list_to_inner(data_list):
    if not data_list:
        return {}

    # Ensure all dictionaries have the same keys
    keys = data_list[0].keys()
    for d in data_list:
        if d.keys() != keys:
            raise ValueError("Not all dicts have the same keys!")

    # Concatenate values for each key
    stacked = {}
    for key in keys:
        stacked[key] = [d[key] for d in data_list]

    # Check that all keys in the stacked have the same length
    lengths = [len(stacked[key]) for key in stacked]
    if len(set(lengths)) != 1:
        raise ValueError("All keys in the stacked dict must have the same length.")

    return stacked


class PHMDMultiSourceLoader(AbstractDataSourceLoader, ABC):
    """This class instantiates each of the soure"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fold: int = kwargs["fold"]
        self.task_mode: str = kwargs["task_mode"]
        self.cache_dir: str = kwargs["cache_dir"]
        self.auxiliary_tasks: list = kwargs.get("auxiliary_tasks", [])

        cache_path = Path(self.cache_dir).expanduser()
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_path)

        self._is_isplitted = False
        self._is_loaded = False
        self.data_dict = None
        self.meta_data = None

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For PHMD, multisource_data_splitter is not supported. PHMD are given with splits."
        self.multisource_data_splitter = "For PHMD, multisource_data_splitter is not supported. PHMD are given with splits."

    def __repr__(self) -> str:
        if self._is_loaded:
            return f"MultiSource(sources={len(self.data_source_dict)} units)"
        return "MultiSource(unloaded)"

    @override
    def load_data(self):
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True

    @override
    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """
        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_isplitted is True, "Data must be splitted before getting data."
        return SplitDatasetContainer(**self.data_dict)

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return self.meta_data

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.data_dict["metadata"]

    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    @override
    def split_data(self):
        logger.warning(
            "PHMD does not support data splitting. The project relies on predefined splits."
        )

    def _init_phmd_dataset(self, task_names):

        ds = datasets.Dataset(self.data_name, cache_dir=self.cache_dir)

        tasks_fold_dict = {}
        for t in task_names:
            task = ds.get_task(t)
            fold = task[self.fold]
            if t == self.task_mode:
                # Extract the main task meta data
                self.meta_data = task.meta

            tasks_fold_dict[t] = fold
        return tasks_fold_dict

    def _cache_task_data(self, task_fold_dict):
        if not self.debug_cache_path:
            raise ValueError("debug_cache_path must be set when caching task data.")

        filename = self._get_cache_file_name("dataset")
        cache_path = Path(self.debug_cache_path) / filename

        logger.info(f"Caching task data to {cache_path}")

        with open(cache_path, "wb") as f:
            pickle.dump(task_fold_dict, f)

        metadata_filename = self._get_cache_file_name("metadata")
        metadata_path = Path(self.debug_cache_path) / metadata_filename

        logger.info(f"Caching task metadata to {metadata_path}")

        with open(metadata_path, "wb") as f:
            pickle.dump(self.meta_data, f)

    def _retrieve_cached_task_data(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")
        else:
            with open(path, "rb") as f:
                return pickle.load(f)

    def _get_cache_file_name(self, data_type) -> str:
        """Generate the cache file name based on the current state."""
        filename_parts = [
            f"taskmode_{self.task_mode}",
            f"fold_{self.fold}",
            f"aux_{'_'.join(self.auxiliary_tasks) if self.auxiliary_tasks else 'none'}",
            f"dataset_{self.data_name}",
            f"type_{data_type}",
        ]
        return "_".join(filename_parts) + ".pkl"

    def _load_data(self):

        tasks_fold_dict = None
        # If we are in debug mode, we try to load the cached data
        if self.debug_cache_dataset:
            cache_path = Path(self.debug_cache_path)
            metadata_file = cache_path / self._get_cache_file_name("metadata")
            task_file = cache_path / self._get_cache_file_name("dataset")

            if metadata_file.exists() and task_file.exists():
                tasks_fold_dict = self._retrieve_cached_task_data(task_file)
                self.meta_data = self._retrieve_cached_task_data(metadata_file)
                logger.info(f"Loaded cached task data from {task_file}")
                logger.info(f"Loaded cached metadata from {metadata_file}")

        if tasks_fold_dict is None:
            tasks_fold_dict = self._init_phmd_dataset(
                self.auxiliary_tasks + [self.task_mode]
            )
            if self.debug_cache_dataset:
                self._cache_task_data(tasks_fold_dict)

        if self.debug_subset_range is not None:
            for key in tasks_fold_dict:
                for subkey in tasks_fold_dict[key]:
                    tasks_fold_dict[key][subkey] = tasks_fold_dict[key][subkey].iloc[
                        range(*self.debug_subset_range)
                    ]

        if self.debug_subsample_rate is not None:
            for key in tasks_fold_dict:
                for subkey in tasks_fold_dict[key]:
                    tasks_fold_dict[key][subkey] = tasks_fold_dict[key][subkey].iloc[
                        :: self.debug_subsample_rate, :
                    ]

        main_fold = tasks_fold_dict[self.task_mode]
        auxiliary_tasks_fold_dict = tasks_fold_dict.copy()
        auxiliary_tasks_fold_dict.pop(self.task_mode, None)
        # TODO: add some clarification on how does the output of _preprocess_fold look like
        main_fold = self._preprocess_fold(main_fold, auxiliary_tasks_fold_dict)

        # Process each split
        data = defaultdict(list)
        unit_col = self._get_unit_column()

        for split in ["train", "val", "test"]:
            df = main_fold[split]
            for unit_name in np.unique(df[unit_col]):

                # We change the masking from this because
                # it looks like in some data frames, df[unit_col] is not of shape
                # (n, ) but (n, 1), creating a 2d mask without failing, which
                # masks out the data (since it acts as row filter as well)
                # df_unit = df[df[unit_col] == unit_name]

                # 1) Are NaNs already present before filtering?
                pre = df.isna().sum()

                # TODO: Check whenever it is a pd.Series or pd.DataFrame
                unit_col_data = df[unit_col]
                if isinstance(unit_col_data, pd.DataFrame):
                    unit_col_data = unit_col_data.squeeze("columns")
                m = pd.Series((unit_col_data == unit_name), index=df.index)
                assert m.dtype == bool and m.ndim == 1 and len(m) == len(df)
                assert not df.columns.duplicated().any(), "Duplicate columns found"
                # TODO: This one breaks the XTJU dataset
                # assert not df.index.duplicated().any(), "Duplicate index found"
                df_unit = df.loc[m]

                # 4) Prove selection itself did not create NaNs
                post = df_unit.isna().sum()
                assert (pre == post).all(), "Selection created new NaNs"

                data[split].append(
                    self._process_unit(
                        df_unit=df_unit,
                        unit_name=unit_name,
                        features_col=self._get_features_columns(),
                        target_col=self._get_target_column(),
                    )
                )

        # The data structure in data is:
        # {
        #     "train": [unit1_data, unit2_data, ...],
        #     "val": [unit1_data, unit2_data, ...],
        #     "test": [unit1_data, unit2_data, ...]
        # }
        # where unitx_data has the structure
        # {
        #     "features": ...,
        #     "timestamps": ...,
        #     "target": ...,
        #     "metadata":
        # }

        # Convert to desired format
        train = convert_outer_list_to_inner(data["train"])
        val = convert_outer_list_to_inner(data["val"])
        test = convert_outer_list_to_inner(data["test"])

        # Now train has the structure
        # {
        #     "features": [unit1_data, unit2_data, ...],
        #     "timestamps": [unit1_data, unit2_data, ...],
        #     "target": [unit1_data, unit2_data, ...],
        #     "metadata": [unit1_data, unit2_data, ...]
        # }

        out_dict = {}
        for key in train.keys():
            out_dict[key] = {
                "train": train[key],
                "val": val[key],
                "test": test[key],
            }

        # Now the structure is
        # {
        #     "features": {"train": [...], "val": [...], "test": [...]},
        #     "timestamps": {"train": [...], "val": [...], "test": [...]},
        #     "target": {"train": [...], "val": [...], "test": [...]},
        #     "metadata": {"train": [...], "val": [...], "test": [...]}
        # }

        logger.info(f"Number of train units: {len(out_dict['features']['train'])}")
        logger.info(f"Number of val units: {len(out_dict['features']['val'])}")
        logger.info(f"Number of test units: {len(out_dict['features']['test'])}")
        return out_dict

    @abstractmethod
    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        raise NotImplementedError("Subclasses must implement _process_unit")

    @abstractmethod
    def _preprocess_fold(self, main_fold, auxiliary_tasks_fold_dict):
        """
        Preprocess the fold for the main task
        """
        raise NotImplementedError("Subclasses must implement _rename_columns")

    @abstractmethod
    def _get_features_columns(self) -> list[str]:
        """Return the list of feature columns."""
        raise NotImplementedError("Subclasses must implement _get_features_columns")

    @abstractmethod
    def _get_target_column(self) -> str:
        """Return the target column name."""
        raise NotImplementedError("Subclasses must implement _get_target_column")

    @abstractmethod
    def _get_unit_column(self) -> str:
        """Return the unit column name."""
        raise NotImplementedError("Subclasses must implement _get_unit_column")
