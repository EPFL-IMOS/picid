"""Single-source datasource loader base.

This module contains the shared lifecycle for datasources that own a single
logical source, such as a bearing or a single machine unit. Concrete loaders
implement only raw loading while this base handles split orchestration,
container normalization, and loader state transitions.
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, Union, cast, override

import numpy as np
from box import Box
import awkward as ak
from pandas.core.generic import NDFrame

from picid.data.data_objects import DatasetContainer, SplitDatasetContainer
from picid.data.datasources.base.contracts import LoaderState
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader

logger = logging.getLogger(__name__)


class SingleSourceLoader(AbstractDataSourceLoader):
    """Shared loader for datasources that expose one logical source.

    Parameters
    ----------
    is_part_of_multisource : bool, optional
        Flag indicating whether the loader is owned by a multisource parent.
        Multisource children do not perform their own split orchestration.
    data_splitter : Any, optional
        Splitter used to partition the datasource when it is managed as a
        standalone single-source loader.
    **kwargs : dict[str, Any]
        Datasource configuration forwarded to
        :class:`~picid.data.datasources.base.interfaces.AbstractDataSourceLoader`.
    """

    def __init__(
        self,
        is_part_of_multisource: bool = False,
        data_splitter: Any = None,
        **kwargs,
    ):
        """Initialise the single-source loader lifecycle state."""
        super().__init__(**kwargs)

        self.is_part_of_multisource = is_part_of_multisource
        self.data_splitter = data_splitter if not is_part_of_multisource else None

        # Keep all loaded payloads in one place so split normalization can work
        # across concrete loaders without additional per-subclass plumbing.
        self.data_dict: Dict[str, Union[np.ndarray, ak.Array, NDFrame]] | Box = {}

        # The flags are created eagerly because some concrete loaders materialize
        # predefined splits during _load_data() instead of via split_data().
        self._is_splitted = False
        self._is_loaded = False

    @abstractmethod
    def _load_data(self) -> Dict[str, Union[np.ndarray, ak.Array, NDFrame]]:
        """Load raw data from the datasource backend.

        Returns
        -------
        dict[str, numpy.ndarray | awkward.Array | pandas.core.generic.NDFrame]
            Mapping of payload names to loaded arrays or frames.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def split_data(self):
        """Split the loaded data unless the loader is owned by a multisource parent.

        Returns
        -------
        None
            The method updates the internal loader state in place.
        """

        if not self.is_part_of_multisource:
            if not self._is_splitted:
                self._require_configuration(
                    self.data_splitter is not None,
                    "Data splitter must be set.",
                )

                # Default to the canonical time-series payloads when the splitter
                # does not explicitly list which keys should be partitioned.
                for key in getattr(
                    self.data_splitter, "create_splits_for", ["features", "timestamps"]
                ):
                    self._split_data(key)
                self._is_splitted = True
                self._state = LoaderState.SPLIT
            else:
                logger.warning("Attempting to split data that has already been split.")

    @override
    def load_data(self):
        """Load the datasource payload and reset any previous split state."""
        self.data_dict = {}
        self._is_loaded = False
        self._is_splitted = False
        self._state = LoaderState.INITIALIZED
        self.data_dict.update(self._load_data())
        self._is_loaded = True
        self._state = LoaderState.LOADED

    def get_data(self) -> SplitDatasetContainer | DatasetContainer:
        """Return the loaded dataset container.

        Returns
        -------
        SplitDatasetContainer | DatasetContainer
            Split-aware container for standalone loaders, or a plain container
            when the loader participates in a multisource composition.
        """

        if not self.is_part_of_multisource:
            return self._get_split_data_container()
        return self._get_loaded_data_container()

    def get_meta_data(self) -> Dict[str, Any]:
        """Return metadata collected by the concrete loader.

        Returns
        -------
        dict[str, Any]
            Loader-specific metadata. The default base implementation returns an
            empty dictionary and concrete loaders can extend it as needed.
        """
        self._require_loaded("getting metadata")
        return {}

    def _get_loaded_data_container(self) -> DatasetContainer:
        """Build a plain dataset container from the loaded payload."""
        self._require_loaded("getting data")
        return DatasetContainer(**dict(self.data_dict))

    def _get_split_data_container(self) -> SplitDatasetContainer:
        """Build a split dataset container from the normalized payload."""
        self._require_loaded("getting data")
        self._require_split_ready("getting data")
        return SplitDatasetContainer(**self._normalize_split_data_dict())

    def get_loaded_data_for_composition(self) -> DatasetContainer:
        """Return the unsplit container used by multisource orchestration."""
        return self._get_loaded_data_container()

    def get_split_data_for_composition(self) -> SplitDatasetContainer:
        """Return the split container used by multisource orchestration."""
        return self._get_split_data_container()

    def _normalize_split_data_dict(self) -> dict[str, Any]:
        """Normalize loaded values into the split-container shape."""
        normalized_data: dict[str, Any] = {}

        for key, value in self.data_dict.items():
            if key == "metadata" and not self._is_split_mapping(value):
                normalized_data["container_metadata"] = value
                continue

            if key == "metadata" and self._is_split_mapping(value):
                normalized_data["unit_metadata"] = {
                    split: split_value
                    if isinstance(split_value, list)
                    else [split_value]
                    for split, split_value in value.items()
                }
                continue

            if self._is_split_mapping(value):
                normalized_data[key] = {
                    split: split_value
                    if isinstance(split_value, list)
                    else [split_value]
                    for split, split_value in value.items()
                }
                continue

            normalized_data[key] = value

        return normalized_data

    @staticmethod
    def _is_split_mapping(value: Any) -> bool:
        """Return True when ``value`` already stores train/val/test entries."""
        if not isinstance(value, dict):
            return False
        split_mapping = cast(dict[str, Any], value)
        return all(split in split_mapping for split in ("train", "val", "test"))

    def _split_data(self, variable) -> Dict[str, np.ndarray]:
        """Split a single payload into train/val/test buckets.

        Parameters
        ----------
        variable : str
            Payload key to split into train, validation, and test partitions.

        Returns
        -------
        dict[str, numpy.ndarray]
            Dictionary returned by the configured splitter for ``variable``.

        Raises
        ------
        DatasourceStateError
            If the datasource has not been loaded before the split is requested.
        """
        self._require_loaded("splitting")

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
            # Preserve the original mask geometry so repeated split variables
            # cannot silently drift from the first split decision.
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
        return splitted_data
