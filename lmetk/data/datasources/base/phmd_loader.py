import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from lmetk.data.datasources.base.interfaces import DataSourceLoaderInterface
from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter
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


class PHMDMultiSourceLoader(DataSourceLoaderInterface, ABC):
    """This class instantiates each of the soure"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fold = kwargs["fold"]
        self.task_mode = kwargs["task_mode"]
        self.cache_dir = kwargs["cache_dir"]

        cache_path = Path(self.cache_dir).expanduser()
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_path)

        # Instantiate multisource_data_splitter if is not none
        assert (
            kwargs.get("multisource_data_splitter") == None
        ), "For PHMD, multisource_data_splitter is not supported. PHMD are given with splits."
        self.multisource_data_splitter = "For PHMD, multisource_data_splitter is not supported. PHMD are given with splits."

    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """

        # Load data from each source
        # Load sources
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True

        return self.data_dict

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return {}

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.data_dict["metadata"]

    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    def split_data(self, data_lst: list[Dict], source_names: list[str]):
        logger.warning(
            "PHMD does not support data splitting. The project relies on predefined splits."
        )

    def _load_data(self):

        ds = datasets.Dataset(self.data_name, cache_dir=self.cache_dir)
        task = ds[self.task_mode]
        meta_data = task.meta
        folds = task[self.fold]  # fold_num = 0 by default

        features_col = meta_data["features"]
        unit_id = meta_data["identifier"]
        assert len(unit_id) == 1, "There should be only one identifier"

        unit_col = unit_id[0]
        target_col = self.task_mode
        features_col = ["mode", "voltage", "current", "temperature"]

        # Process each split
        data = defaultdict(list)
        for split in ["train", "val", "test"]:
            df = folds[split]
            for unit_name in np.unique(df[unit_col]):
                df_unit = df[df[unit_col] == unit_name]
                data[split].append(
                    self._process_unit(
                        df_unit=df_unit,
                        unit_name=unit_name,
                        features_col=features_col,
                        target_col=target_col,
                    )
                )

        # Convert to desired format
        train = convert_outer_list_to_inner(data["train"])
        val = convert_outer_list_to_inner(data["val"])
        test = convert_outer_list_to_inner(data["test"])

        out_dict = {}
        for key in train.items():
            out_dict[key] = {
                "train": train[key],
                "val": val[key],
                "test": test[key],
            }

        logger.info(f"Number of train units: {len(out_dict['features']['train'])}")
        logger.info(f"Number of val units: {len(out_dict['features']['val'])}")
        logger.info(f"Number of test units: {len(out_dict['features']['test'])}")
        return out_dict

    @abstractmethod
    def _process_unit(self, df_unit, unit_name, features_col, target_col):
        raise NotImplementedError("Subclasses must implement _process_unit")

    @abstractmethod
    def _rename_columns(self, folds):
        raise NotImplementedError("Subclasses must implement _rename_columns")
