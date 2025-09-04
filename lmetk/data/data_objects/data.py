import copy
import warnings
import logging
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd

from lmetk.data.data_objects.base import (
    BaseDataObject,
)  # Assuming your BaseDataObject is here
from lmetk.data.data_objects.utils import (
    check_for_nans,
    check_length_consistency,
    convert_to_numpy,
)

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class DataChunk(BaseDataObject):
    """
    A flexible data container that stores an individual data unit in its
    original format.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the chunk, handling metadata and column extraction robustly.
        """
        # --- FIX ---
        # Pop metadata to handle it separately after other items are initialized.
        # This avoids order-of-operation bugs.
        metadata = kwargs.pop("metadata", None)

        # Initialize all other data first
        super().__init__(**kwargs)

        # Now, if metadata was provided, prepare it and set it.
        if metadata is not None:
            # Ensure column_map exists in the user's provided metadata
            metadata.setdefault("column_map", {})
            # Now that `self` is populated, find all DataFrames and update the map
            for key, value in self.items():
                if isinstance(value, pd.DataFrame):
                    metadata["column_map"][key] = value.columns.tolist()

            # Finally, set the fully prepared metadata object
            self["metadata"] = metadata

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets an item, storing it in its original format and intelligently
        updating the column_map.
        """
        # --- FIX ---
        # Call the parent's (BaseDataObject's) setter first.
        # This handles the recursive logic for non-metadata keys.
        super().__setitem__(key, value)

        # After setting, manage the column map if metadata exists.
        if "metadata" in self and isinstance(self.metadata, dict):
            is_df = isinstance(value, pd.DataFrame)
            self.metadata.setdefault("column_map", {})

            if is_df:
                self.metadata["column_map"][key] = value.columns.tolist()
            elif key in self.metadata.get("column_map", {}):
                # If a key that had columns is being replaced by a non-DataFrame,
                # remove its stale entry from the map.
                del self.metadata["column_map"][key]

    # --- Public API Methods ---

    def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
        """
        Converts the data for a given key to a NumPy array on demand.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DataChunk.")
        value = self[key]
        if isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            dtypes = getattr(value, "dtypes", [getattr(value, "dtype", None)])
            if any(dt.kind not in "biufc" for dt in dtypes):
                array = value.values
            else:
                array = value.values.astype(np.float32)
        else:
            array = np.array(value)
        if ensure_2d:
            if array.ndim == 1:
                return array.reshape(-1, 1)
            if array.ndim != 2:
                raise ValueError(
                    f"Key '{key}' has unsupported dimensions: {array.ndim}."
                )
        return array

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame representation of the data for a given key.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DataChunk.")
        value = self[key]
        if isinstance(value, pd.DataFrame):
            return value
        array_data = self.to_numpy(key, ensure_2d=False)
        columns = self.get("metadata", {}).get("column_map", {}).get(key)
        return pd.DataFrame(array_data, columns=columns)

    def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
        """
        Performs length and NaN consistency checks using external utilities.
        """
        keys = keys_to_validate if keys_to_validate is not None else self.keys()
        items_to_check = {k: self[k] for k in keys if k in self}

        if not items_to_check:
            logging.info("No items to validate.")
            return

        # Perform validation checks by calling the imported utility functions
        values = list(items_to_check.values())
        keys = list(items_to_check.keys())

        check_for_nans(values, keys)
        check_length_consistency(values, keys)

        logging.info("Validation successful for keys: %s", keys)


ARRAY_LIKE_TYPES = (pd.DataFrame, pd.Series, np.ndarray, list, tuple)


class OrderedInput(BaseDataObject):
    """
    A data object that holds inputs for a transformation and can sanitize them.
    It includes methods for converting its contents to NumPy or DataFrame formats.
    """

    # --- Public API Methods ---

    def get_sanitized_data(
        self,
        check_nans: bool = True,
        check_lengths: bool = True,
        warn_only: bool = False,
        size_threshold: int = 1_000_000,
        return_copy: bool = False,
    ) -> "OrderedInput":
        """
        Performs validation, extracts column metadata, and converts data to NumPy arrays.
        """
        target_obj = copy.deepcopy(self) if return_copy else self

        # # Replace .setdefault() with standard assignment to trigger our custom __setitem__
        # if any(isinstance(v, pd.DataFrame) for v in target_obj.values()) and (
        #     "metadata" in target_obj
        # ):
        #     # Do the same for the nested 'column_map'.
        #     if "column_map" not in target_obj.metadata:
        #         target_obj.metadata["column_map"] = {}

        #     # Populate the column map. This chained access now works correctly.
        #     for key, value in target_obj.items():
        #         if isinstance(value, pd.DataFrame):
        #             target_obj.metadata.column_map[key] = value.columns.tolist()

        # # --- 1. CONVERT ALL ARRAY-LIKE DATA TO NUMPY ---
        # for key, value in list(target_obj.items()):
        #     if isinstance(value, ARRAY_LIKE_TYPES):
        #         target_obj[key] = convert_to_numpy(value, key)

        # --- 2. PERFORM CONDITIONAL CHECKS on the converted NumPy arrays ---
        values = list(target_obj.values())
        keys = list(target_obj.keys())

        if check_nans:
            check_for_nans(values, keys, raise_on_error=(not warn_only))

        if check_lengths:
            total_size = sum(v.size for v in values if hasattr(v, "size"))
            if total_size < size_threshold:
                check_length_consistency(values, keys, raise_on_error=(not warn_only))
            else:
                warnings.warn(
                    f"Data size ({total_size}) exceeds threshold ({size_threshold}). "
                    "Skipping length consistency check for performance."
                )

        return target_obj

    def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
        """
        Converts the data for a given key to a NumPy array on demand.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in OrderedInput.")
        value = self[key]
        # This calls the standalone utility function we created earlier
        return convert_to_numpy(value, key, ensure_2d)

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame representation of the data for a given key.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in OrderedInput.")

        value = self[key]
        if isinstance(value, pd.DataFrame):
            return value

        array_data = self.to_numpy(key, ensure_2d=False)
        # It safely checks for metadata, which may or may not exist on an OrderedInput
        columns = self.get("metadata", {}).get("column_map", {}).get(key)
        return pd.DataFrame(array_data, columns=columns)


"""
# --- Return Object Types ---
We provide several specialized return object types to standardize outputs from transformations.
However, we had no time to implement them fully, so they currently do not add functionality beyond BaseDataObject.
"""


class SimpleReturnObject(BaseDataObject):
    """A data object for simple, single-array returns from a transform."""

    pass


class NamedDictReturnObject(BaseDataObject):
    """A data object for dictionary-like returns with named keys."""

    pass


class ExtendedReturnObject(BaseDataObject):
    """A data object for complex returns that may include data and metadata."""

    pass


class DatasetContainer(BaseDataObject):
    """
    A container for a complete dataset composed of multiple splits and units.

    It stores data grouped by type (e.g., features, targets), automatically
    normalizes single-unit data into lists, and provides methods for validation
    and transformation.
    """

    # --- Dunder Methods ---

    def __init__(self, *args, is_multi_unit: bool = True, **kwargs: Any) -> None:
        """
        Initializes the chunk, normalizing all split data to be a list of units.
        """
        # Assume multi-unit by default. If we find any single item, we flip the flag.
        self.is_multi_unit: bool = is_multi_unit

        # check if every element in args is in fact also a. DatasetContainer
        # merge them into one
        # also make sure that kwargs is empty when args has content and otherwise
        # also make sure that keys of args are the same

        # Normalize the input data to ensure unit data is always in a list
        for data_key, splits_dict in kwargs.items():
            if isinstance(splits_dict, dict):
                for split, data_for_split in splits_dict.items():
                    # If the data for a split is not a list, wrap it in one
                    if not isinstance(data_for_split, list):
                        logging.info(
                            f"Found single item for '{data_key}/{split}'. "
                            "Wrapping it in a list for consistency."
                        )
                        self.is_multi_unit = False  # Found a single-unit entry
                        splits_dict[split] = [data_for_split]

        # The base __init__ will now receive the normalized kwargs
        super().__init__(**kwargs)

    # --- Public API Methods ---

    def validate(self) -> None:
        """
        Performs comprehensive integrity checks on the dataset structure.

        This method verifies two critical rules:
        1.  For each split, all feature lists must contain the same number of units.
        2.  For each split, every unit must have the exact same set of feature keys.

        Raises:
            ValueError: If any validation check fails.
        """
        if not self:
            logging.info("MultiDatasetChunk is empty. Nothing to validate.")
            return


class SplitDatasetContainer(DatasetContainer):

    def __init__(self, *args, is_multi_unit=True, **kwargs):
        super().__init__(*args, is_multi_unit=is_multi_unit, **kwargs)

    def validate(self):
        super().validate()

        all_splits = set()
        for splits_data in self.values():
            if isinstance(splits_data, dict):
                all_splits.update(splits_data.keys())

        for split in sorted(list(all_splits)):
            # Check 1: Consistent NUMBER of units across all feature keys for this split
            split_list_lengths = {
                len(self[key][split])
                for key in self
                if isinstance(self[key], dict) and split in self[key]
            }
            if len(split_list_lengths) > 1:
                raise ValueError(
                    f"Inconsistent number of units for split '{split}'. "
                    f"Found lists with differing lengths: {split_list_lengths}"
                )

            num_units = split_list_lengths.pop() if split_list_lengths else 0
            if num_units == 0:
                continue

            # Check 2: Consistent FEATURE KEYS within each unit for this split
            unit_key_sets = [
                {
                    key
                    for key in self
                    if isinstance(self[key], dict) and split in self[key]
                }
                for i in range(num_units)
            ]

            if unit_key_sets:
                first_key_set = unit_key_sets[0]
                if not all(s == first_key_set for s in unit_key_sets):
                    raise ValueError(
                        f"Inconsistent feature keys for units in split '{split}'. "
                        f"All units in a split must have the same set of data keys. Found: {unit_key_sets}"
                    )

        logging.info(
            "MultiDatasetChunk validation successful. All splits and units are consistent."
        )

    def to_split_dict(self) -> Dict[str, Dict[str, List[Any]]]:
        """
        Reorganizes the data into a dictionary grouped by split.

        This method returns a dictionary where each split contains the original
        lists of feature data, rather than lists of DataChunk objects.

        Returns:
            A dictionary where keys are splits ('train', 'val', 'test')
            and values are dictionaries of feature lists.
        """
        if not self:
            return {}

        all_splits = set()
        for splits_data in self.values():
            if isinstance(splits_data, dict):
                all_splits.update(splits_data.keys())

        split_dict = {split: {} for split in all_splits}
        for data_key, splits_data in self.items():
            if isinstance(splits_data, dict):
                for split, unit_list in splits_data.items():
                    if not self.is_multi_unit:
                        assert (
                            len(unit_list) == 1
                        ), "The number of units is greater than 1, although the data is set to be not multi_unit"
                        split_dict[split][data_key] = unit_list[0]
                    else:
                        split_dict[split][data_key] = unit_list

        return split_dict

    def group_by_split(self) -> Dict[str, List[DataChunk]]:
        """
        Reorganizes the data into a dictionary of DataChunk object lists, grouped by split.

        It is recommended to run .validate() first to ensure data integrity.

        Returns:
            A dictionary where keys are the splits ('train', 'val', 'test')
            and values are lists of DataChunk objects, one for each unit.
        """
        if not self:
            return {}

        splits, num_units_per_split, data_keys = set(), {}, list(self.keys())
        for key in data_keys:
            if isinstance(self[key], dict):
                splits.update(self[key].keys())

        for split in splits:
            lengths = {
                len(self[key][split])
                for key in data_keys
                if isinstance(self[key], dict) and split in self[key]
            }
            if lengths:
                num_units_per_split[split] = lengths.pop()

        grouped_dict: Dict[str, List["DataChunk"]] = {split: [] for split in splits}
        for split in splits:
            num_units = num_units_per_split.get(split, 0)
            for unit_idx in range(num_units):
                unit_data = {
                    data_key: self[data_key][split][unit_idx]
                    for data_key in data_keys
                    if isinstance(self[data_key], dict) and split in self[data_key]
                }
                if unit_data:
                    grouped_dict[split].append(DataChunk(**unit_data))
        return grouped_dict

    def __repr__(self):
        return "MOVi dataset bug fix"
