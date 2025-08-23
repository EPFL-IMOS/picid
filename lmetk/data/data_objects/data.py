import logging
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
from lmetk.data.data_objects.base import BaseDataObject # Assuming your BaseDataObject is here


# Basic logger configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DataChunk(BaseDataObject):
    """
    A flexible data container that stores an individual data unit (e.g., for one
    operational cycle) in its original format.

    This class uses a "Lazy Conversion" model. Data is stored as-is (e.g.,
    as a pandas.DataFrame). Conversion to NumPy and validation are handled
    explicitly through public methods.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the chunk. If metadata is provided, it automatically
        extracts column names from any pandas.DataFrame objects.
        """
        # --- CHANGE STARTS HERE ---
        # Only attempt to extract columns if metadata is already present in the input
        if 'metadata' in kwargs and isinstance(kwargs.get('metadata'), dict):
            # Ensure the column_map sub-dictionary exists within the provided metadata
            kwargs['metadata'].setdefault('column_map', {})
            
            # Extract column names for any DataFrames found
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    kwargs['metadata']['column_map'][key] = value.columns.tolist()
        # --- CHANGE ENDS HERE ---

        # Call the parent's __init__ to handle the actual setting of items
        super().__init__(**kwargs)

    # --- Public API Methods ---

    def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
        """
        Converts the data for a given key to a NumPy array on demand.

        Args:
            key (str): The key of the item to convert.
            ensure_2d (bool): If True, ensures the output array is 2-dimensional.

        Returns:
            A NumPy array representation of the data.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DataChunk.")

        value = self[key]

        # --- Conversion Logic ---
        if isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            # Check dtypes to decide if a safe object array or a numeric array should be created
            dtypes = getattr(value, 'dtypes', [getattr(value, 'dtype', None)])
            if any(dt.kind not in 'biufc' for dt in dtypes): # b:bool, i:int, u:uint, f:float, c:complex
                array = value.values
            else:
                array = value.values.astype(np.float32)
        else:  # For lists, tuples, etc.
            array = np.array(value)

        # --- Dimension Check ---
        if ensure_2d:
            if array.ndim == 1:
                return array.reshape(-1, 1)
            if array.ndim != 2:
                raise ValueError(f"Key '{key}' has unsupported dimensions: {array.ndim}.")
        return array

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame representation of the data for a given key.

        If the item is already a DataFrame, it is returned directly to preserve it.
        Otherwise, it is converted using the stored column map if available.

        Args:
            key (str): The key of the item to retrieve or convert.

        Returns:
            A pandas.DataFrame.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DataChunk.")
        
        value = self[key]
        # If it's already a DataFrame, return it as-is
        if isinstance(value, pd.DataFrame):
            return value

        # Otherwise, convert it to a DataFrame
        array_data = self.to_numpy(key, ensure_2d=False)
        columns = self.get("metadata", {}).get("column_map", {}).get(key)
        return pd.DataFrame(array_data, columns=columns)

    def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
        """
        Performs length and NaN consistency checks on the raw data.

        This is an explicit action. It should be called when you need to ensure
        the consistency of different data arrays within the chunk.

        Args:
            keys_to_validate (Optional[List[str]]): A specific list of keys to validate.
                If None, all items in the chunk will be checked.
        """
        keys = keys_to_validate if keys_to_validate is not None else self.keys()
        items_to_check = {k: self[k] for k in keys if k in self}
        
        if not items_to_check:
            logging.info("No items to validate.")
            return

        # Perform the actual validation checks
        self._check_for_nans(list(items_to_check.values()), list(items_to_check.keys()))
        self._check_length_consistency(list(items_to_check.values()), list(items_to_check.keys()))
        logging.info("Validation successful for keys: %s", list(items_to_check.keys()))

    # --- Dunder Methods ---

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets an item, storing it in its original format and intelligently
        updating the column_map metadata if a DataFrame is added or replaced.
        """
        is_df = isinstance(value, pd.DataFrame)
        
        # --- CHANGE STARTS HERE ---
        # Only manage the column_map if the 'metadata' attribute already exists on the object
        if 'metadata' in self and isinstance(self.metadata, dict):
            self.metadata.setdefault('column_map', {})
            
            if is_df:
                # If the new value is a DataFrame, update its column map
                self.metadata['column_map'][key] = value.columns.tolist()
            elif key in self.metadata.get('column_map', {}):
                # If the key previously had a column map but is being replaced
                # by a non-DataFrame, remove the stale entry.
                del self.metadata['column_map'][key]
        # --- CHANGE ENDS HERE ---

        # Use the grandparent's __setitem__ to store the raw value,
        # bypassing the recursive conversion in BaseDataObject.
        super(BaseDataObject, self).__setitem__(key, value)

    # --- Internal Validation Helpers ---

    def _get_length(self, value: Any) -> Optional[int]:
        """Helper to safely get the length of different array-like types."""
        if hasattr(value, 'shape') and hasattr(value.shape, '__len__') and value.shape:  # Pandas, Numpy
            return value.shape[0]
        if isinstance(value, (list, tuple)):
            return len(value)
        return None  # Return None for non-array-like types

    def _check_length_consistency(self, values: List[Any], keys: List[str]) -> None:
        """Checks that all provided array-like items have the same length."""
        # Get lengths of all items that have a measurable length
        lengths = {self._get_length(v) for v in values if self._get_length(v) is not None}
        if len(lengths) > 1:
            length_info = {k: self._get_length(v) for k, v in zip(keys, values)}
            raise ValueError(f"Length mismatch between items: {length_info}")

    def _check_for_nans(self, values: List[Any], keys: List[str]) -> None:
        """Checks for NaN values in pandas or numpy objects."""
        for value, key in zip(values, keys):
            has_nan = False
            if isinstance(value, pd.DataFrame):
                # Check only numeric columns for NaNs
                numeric_data = value.select_dtypes(include=np.number)
                if not numeric_data.empty:
                    has_nan = numeric_data.isnull().values.any()
            elif isinstance(value, pd.Series):
                # Check Series only if it's a numeric type
                if pd.api.types.is_numeric_dtype(value.dtype):
                    has_nan = value.isnull().any()
            elif isinstance(value, np.ndarray):
                # Check array only if it's a numeric type
                if np.issubdtype(value.dtype, np.number):
                    has_nan = np.isnan(value).any()
            
            if has_nan:
                raise ValueError(f"NaN values found in object '{key}'.")





class DatasetContainer(BaseDataObject):
    """
    A container for a complete dataset composed of multiple splits and units.

    It stores data grouped by type (e.g., features, targets), automatically
    normalizes single-unit data into lists, and provides methods for validation
    and transformation.
    """

    # --- Dunder Methods ---

    def __init__(self, is_multi_unit: bool=True, **kwargs: Any) -> None:
        """
        Initializes the chunk, normalizing all split data to be a list of units.
        """
        # Assume multi-unit by default. If we find any single item, we flip the flag.
        self.is_multi_unit: bool = is_multi_unit

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
                        self.is_multi_unit = False # Found a single-unit entry
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

        all_splits = set()
        for splits_data in self.values():
            if isinstance(splits_data, dict):
                all_splits.update(splits_data.keys())

        for split in sorted(list(all_splits)):
            # Check 1: Consistent NUMBER of units across all feature keys for this split
            split_list_lengths = {len(self[key][split]) for key in self if isinstance(self[key], dict) and split in self[key]}
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
                {key for key in self if isinstance(self[key], dict) and split in self[key]}
                for i in range(num_units)
            ]
            
            if unit_key_sets:
                first_key_set = unit_key_sets[0]
                if not all(s == first_key_set for s in unit_key_sets):
                    raise ValueError(
                        f"Inconsistent feature keys for units in split '{split}'. "
                        f"All units in a split must have the same set of data keys. Found: {unit_key_sets}"
                    )
        
        logging.info("MultiDatasetChunk validation successful. All splits and units are consistent.")

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
            lengths = {len(self[key][split]) for key in data_keys if isinstance(self[key], dict) and split in self[key]}
            if lengths:
                num_units_per_split[split] = lengths.pop()

        grouped_dict: Dict[str, List['DataChunk']] = {split: [] for split in splits}
        for split in splits:
            num_units = num_units_per_split.get(split, 0)
            for unit_idx in range(num_units):
                unit_data = {
                    data_key: self[data_key][split][unit_idx]
                    for data_key in data_keys if isinstance(self[data_key], dict) and split in self[data_key]
                }
                if unit_data:
                    grouped_dict[split].append(DataChunk(**unit_data))
        return grouped_dict