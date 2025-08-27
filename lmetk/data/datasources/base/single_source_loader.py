import logging
from abc import abstractmethod
from typing import Any, Dict, override

import numpy as np
from box import Box

from lmetk.data.data_objects.data import DatasetContainer, SplitDatasetContainer
from lmetk.data.datasources.base.interfaces import AbstractDataSourceLoader

logger = logging.getLogger(__name__)


class SingleSourceLoader(AbstractDataSourceLoader):
    """
    SingleSourceLoader is a data source loader that handles loading data from a single source.
    A source in the context of PHM is most likely a unit, such as a bearing.

    This class implements the DataSourceLoaderInterface and provides methods to load, preprocess,
    and manage data from a specified source. It is intended to be subclassed or instantiated
    with specific source configurations.
    """

    def __init__(
        self,
        is_part_of_multisource: bool = False,
        data_splitter: Any = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_part_of_multisource = is_part_of_multisource
        self.data_splitter = data_splitter if not is_part_of_multisource else None

        # Define data source loader storage dict as centralized storage for the loaded data.
        self.data_dict: Dict[str, Dict[str, np.ndarray]] | Box = {}

        # Split flag, create before load, as some datasets might have predefined splits,
        # hence then _load_data method has to update self._is_splitted flag
        self._is_splitted = False
        self._is_loaded = False

    @abstractmethod
    def _load_data(self) -> Dict[str, np.ndarray]:
        """Load data from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    def split_data(self):
        """
        Splits the data into predefined subsets using the configured data splitter.
        This method checks if the data is part of a multisource dataset and whether it has already been split.
        If not already split, it asserts that a data splitter is set, then creates splits for the specified keys
        (defaulting to "features" and "timestamps" if not specified in the data splitter). After splitting, it marks
        the data as split. If the data has already been split, a warning is logged.
        Raises:
            AssertionError: If the data splitter is not set when attempting to split.
        """

        if not self.is_part_of_multisource:
            if not self._is_splitted:
                assert self.data_splitter is not None, "Data splitter must be set."

                # By default create splits for features and timestamps
                for key in getattr(
                    self.data_splitter, "create_splits_for", ["features", "timestamps"]
                ):
                    self._split_data(key)
                self._is_splitted = True
            else:
                logger.warning("Attempting to split data that has already been split.")

    @override
    def load_data(self):
        self.data_dict.update(self._load_data())
        self._is_loaded = True

    def get_data(self) -> SplitDatasetContainer | DatasetContainer:
        """Return splitted data dict."""

        assert self._is_loaded is True, "Data must be loaded before getting data."

        if not self.is_part_of_multisource:
            assert (
                self._is_splitted is True
            ), "Data must be splitted before getting data."

            # Ensure metadata dict has structure as other dicts
            # {metadata: 'train':value, 'val':value, 'test':value}
            if "metadata" in self.data_dict and not (
                isinstance(self.data_dict["metadata"], dict)
                and all(
                    k in self.data_dict["metadata"] for k in ["train", "val", "test"]
                )
            ):
                # Replicate metadata for each split if not already split
                metadata = self.data_dict["metadata"]
                self.data_dict["metadata"] = {
                    "train": metadata,
                    "val": metadata,
                    "test": metadata,
                }
            return SplitDatasetContainer(**self.data_dict)
        else:
            return DatasetContainer(**self.data_dict)

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return {}

    def get_data_name(self):
        """Return data name."""
        return self.data_name

    def _split_data(self, variable) -> Dict[str, np.ndarray]:
        assert self._is_loaded, "Data must be loaded before splitting."

        splitted_data, split_masks = self.data_splitter.split_data(
            data_dict=self.data_dict.copy(),
            split_variable=variable,
        )

        if not self.data_dict.get("split_masks", None):
            self.data_dict.update(
                {
                    "split_masks": {
                        "train": split_masks["train"],
                        "val": split_masks["val"],
                        "test": split_masks["test"],
                    }
                }
            )
        else:
            # Check that splits masks are consistent
            for key in split_masks.keys():
                assert (
                    splitted_data[key].shape[0]
                    == self.data_dict["split_masks"][key].shape[0]
                ), f"Mismatch in {key} data and timestamps length"

        self.data_dict.update(
            {
                variable: {
                    "train": splitted_data["train"],
                    "val": splitted_data["val"],
                    "test": splitted_data["test"],
                }
            }
        )
