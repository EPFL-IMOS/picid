import logging
from typing import Any, List, Optional
import numpy as np
import pandas as pd

# Basic logger configuration for demonstration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Base Class ---
class AttributeDict(dict):
    """A container to store state variables of your program."""
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delattr__(self, item: str) -> None:
        if item not in self:
            raise KeyError(item)
        del self[item]

    def __repr__(self) -> str:
        if not self:
            return ""
        max_key_length = max(len(str(k)) for k in self)
        rows = []
        for key in sorted(self.keys()):
            value = self[key]
            if isinstance(value, np.ndarray):
                value_repr = f"array(shape={value.shape}, dtype={value.dtype})"
            else:
                value_repr = repr(value)
            rows.append(f'"{key}":'.ljust(max_key_length + 4) + value_repr)
        return "\n".join(rows)


class DatasetChunk(AttributeDict):
    """
    A specialized data container for PHM that handles mixed data types,
    separating array-like data from metadata and auto-tracking DataFrame columns.
    """
    ARRAY_LIKE_TYPES = (pd.DataFrame, pd.Series, np.ndarray, list, tuple)

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the DatasetChunk and validates the input data."""
        super().__init__()

        # Ensure metadata dictionary exists if needed
        if any(isinstance(v, pd.DataFrame) for v in kwargs.values()):
            kwargs.setdefault('metadata', {})
            kwargs['metadata'].setdefault('column_map', {})

        # Automatically extract column names from ANY DataFrame
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                logging.info(f"Extracting columns for DataFrame '{key}' into metadata.column_map.")
                kwargs['metadata']['column_map'][key] = value.columns.tolist()

        for key, value in kwargs.items():
            self[key] = value
        

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets an item, performing efficient, incremental validation for arrays.
        """
        if isinstance(value, self.ARRAY_LIKE_TYPES):
            # 1. Process the new value first
            processed_value = self._convert_to_numpy(value, key)
            processed_value = self._ensure_2d(processed_value, key)

            # 2. Perform efficient, local checks BEFORE adding the item
            # Check for NaNs only in the new array
            self._check_for_nans([processed_value], [key])

            # # Check for length consistency against existing arrays
            # existing_arrays = [
            #     v for v in self.values() if isinstance(v, np.ndarray)
            # ]
            # if existing_arrays:
            #     expected_length = existing_arrays[0].shape[0]
            #     new_length = processed_value.shape[0]
            #     if new_length != expected_length:
            #         logging.info(
            #             f"Length mismatch for key '{key}'. "
            #             f"Expected length {expected_length}, but got {new_length}."
            #         )

            # 3. If all local checks pass, add the item
            super().__setitem__(key, processed_value)
        else:
            # For non-array types like metadata, just add them
            logging.info(f"Storing key '{key}' as-is (type: {type(value).__name__}).")
            super().__setitem__(key, value)

    def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
        """
        Performs consistency checks on specified numpy arrays, or all arrays if none are specified.

        Args:
            keys_to_validate (Optional[List[str]]): A list of keys to validate.
                If None, all numpy arrays in the chunk will be validated.
        """
        all_array_items = {
            k: v for k, v in self.items() if isinstance(v, np.ndarray)
        }

        if keys_to_validate is None:
            # If no keys are specified, validate all arrays
            items_to_check = all_array_items
        else:
            # Otherwise, filter for the specified keys that are actually arrays
            items_to_check = {
                k: all_array_items[k]
                for k in keys_to_validate
                if k in all_array_items
            }

        if not items_to_check:
            logging.info("No arrays to validate.")
            return

        keys = list(items_to_check.keys())
        arrays = list(items_to_check.values())

        self._check_length_consistency(arrays, keys)
        self._check_for_nans(arrays, keys)
        logging.info("Array validation successful for keys: %s", keys)

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Converts a specified numpy array back to a pandas DataFrame using
        its stored column names from `metadata.column_map`.

        Args:
            key (str): The key of the array to convert (e.g., 'features').
        """
        if key not in self or not isinstance(self[key], np.ndarray):
            raise AttributeError(f"Cannot create DataFrame: key '{key}' with a numpy array not found.")

        array_data = self[key]
        columns = self.get("metadata", {}).get("column_map", {}).get(key)

        if columns and len(columns) != array_data.shape[1]:
            logging.warning(
                f"Column count mismatch for '{key}': Metadata has {len(columns)} names, "
                f"but array has {array_data.shape[1]} columns. Ignoring names."
            )
            columns = None

        return pd.DataFrame(array_data, columns=columns)

    # --- Internal Helper Methods (Unchanged) ---
    def _convert_to_numpy(self, value: Any, key: str) -> np.ndarray:
        if isinstance(value, (pd.DataFrame, pd.Series)):
            logging.info(f"Converting pandas object '{key}' to a numpy array.")
            return value.values.astype(np.float32)
        if isinstance(value, np.ndarray): return value
        try:
            array = np.array(value, dtype=np.float32)
            logging.info(f"Converted list/tuple '{key}' to a numpy array.")
            return array
        except Exception as e:
            raise TypeError(f"Could not convert key '{key}' to a numpy array: {e}") from e

    def _ensure_2d(self, value: np.ndarray, key: str) -> np.ndarray:
        if value.ndim == 1: return value.reshape(-1, 1)
        if value.ndim != 2: raise ValueError(f"Key '{key}' has unsupported dimensions: {value.ndim}.")
        return value

    def _check_length_consistency(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        if not arrays: return
        lengths = {arr.shape[0] for arr in arrays}
        if len(lengths) > 1:
            length_info = {k: v.shape[0] for k, v in zip(keys, arrays)}
            raise ValueError(f"Length mismatch between arrays: {length_info}")

    def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        for arr, key in zip(arrays, keys):
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN values found in array '{key}'")