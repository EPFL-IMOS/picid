import logging
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from lmetk.data.data_objects.base import BaseDataObject

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DatasetChunk(BaseDataObject):
    """
    A specialized data container for holding and processing array-like PHM data.

    It automatically converts pandas objects and lists into NumPy arrays while
    preserving metadata and tracking original DataFrame columns. Length consistency
    must be checked explicitly by calling the .validate() method.
    """
    ARRAY_LIKE_TYPES = (pd.DataFrame, pd.Series, np.ndarray, list, tuple)

    # --- Public API Methods ---

    def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
        """
        Performs length and NaN consistency checks on specified numpy arrays.

        If no keys are specified, all numpy arrays in the chunk will be validated.
        This is the explicit way to check for length mismatches.

        Args:
            keys_to_validate (Optional[List[str]]): A list of keys to validate.
        """
        all_array_items = {k: v for k, v in self.items() if isinstance(v, np.ndarray)}
        items_to_check = all_array_items if keys_to_validate is None else {
            k: all_array_items[k] for k in keys_to_validate if k in all_array_items
        }

        if not items_to_check:
            logging.info("No arrays to validate.")
            return

        keys = list(items_to_check.keys())
        arrays = list(items_to_check.values())

        # Perform all validations here
        self._check_length_consistency(arrays, keys)
        self._check_for_nans(arrays, keys) # Note: _check_for_nans on arrays is faster
        logging.info("Full validation successful for keys: %s", keys)

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Converts a specified numpy array back to a pandas DataFrame.
        """
        if key not in self or not isinstance(self[key], np.ndarray):
            raise AttributeError(f"Cannot create DataFrame: key '{key}' with a numpy array not found.")
        array_data, columns = self[key], self.get("metadata", {}).get("column_map", {}).get(key)
        if columns and len(columns) != array_data.shape[1]:
            logging.warning(f"Column count mismatch for '{key}': Metadata has {len(columns)} names, but array has {array_data.shape[1]} columns. Ignoring names.")
            columns = None
        return pd.DataFrame(array_data, columns=columns)

    # --- Dunder Methods ---

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the chunk and handles DataFrame column extraction."""
        # Auto-extract column names from ANY DataFrame into a nested metadata map
        if any(isinstance(v, pd.DataFrame) for v in kwargs.values()):
            kwargs.setdefault('metadata', {})
            kwargs['metadata'].setdefault('column_map', {})
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                logging.info(f"Extracting columns for DataFrame '{key}' into metadata.column_map.")
                kwargs['metadata']['column_map'][key] = value.columns.tolist()
        
        # Call the parent's __init__ to handle the actual setting of items
        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets an item, converting array-likes and preserving metadata."""
        if isinstance(value, self.ARRAY_LIKE_TYPES):
            # Perform pre-conversion NaN check on the raw object
            self._check_for_nans_pre_conversion([value], [key])
            # Process the value
            processed_value = self._convert_to_numpy(value, key)
            processed_value = self._ensure_2d(processed_value, key)
            super().__setitem__(key, processed_value)
        else:
            # For non-array types, just store them
            logging.info(f"Storing key '{key}' as-is (type: {type(value).__name__}).")
            super().__setitem__(key, value)

    # --- Internal Validation Helpers ---

    def _check_length_consistency(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        """Checks that all provided numpy arrays have the same number of rows."""
        if not arrays: return
        lengths = {arr.shape[0] for arr in arrays}
        if len(lengths) > 1:
            length_info = {k: v.shape[0] for k, v in zip(keys, arrays)}
            raise ValueError(f"Length mismatch between arrays: {length_info}")

    def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
        """Checks for NaNs in numpy arrays (more efficient for validation)."""
        for arr, key in zip(arrays, keys):
            if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
                raise ValueError(f"NaN values found in array '{key}'.")

    def _check_for_nans_pre_conversion(self, values: List[Any], keys: List[str]) -> None:
        """Checks for NaNs in various raw data types before conversion."""
        for value, key in zip(values, keys):
            has_nan = False
            # Handle pandas DataFrames
            if isinstance(value, pd.DataFrame):
                numeric_data = value.select_dtypes(include=np.number)
                if not numeric_data.empty:
                    has_nan = numeric_data.isnull().values.any()

            # --- FIX STARTS HERE ---
            # Handle pandas Series separately
            elif isinstance(value, pd.Series):
                # A Series has one dtype, so check if it's numeric before looking for NaNs
                if pd.api.types.is_numeric_dtype(value.dtype):
                    has_nan = value.isnull().any()
            # --- FIX ENDS HERE ---

            # Handle numpy arrays
            elif isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.number):
                    has_nan = np.isnan(value).any()

            # Handle lists/tuples
            elif isinstance(value, (list, tuple)):
                try:
                    numeric_list = [x for x in value if isinstance(x, (int, float))]
                    if numeric_list and np.isnan(np.array(numeric_list)).any():
                        has_nan = True
                except TypeError:
                    pass
            
            if has_nan:
                raise ValueError(f"NaN values found in object '{key}' before conversion.")

    # --- Internal Processing Helpers ---

    def _convert_to_numpy(self, value: Any, key: str) -> np.ndarray:
        """
        Converts array-like objects to a numpy array with an appropriate dtype.
        """
        if isinstance(value, np.ndarray):
            return value

        # --- FIX STARTS HERE ---
        # For pandas objects, inspect dtypes to decide conversion strategy
        if isinstance(value, pd.DataFrame):
            # If there are any non-numeric types, use 'object' dtype for safety
            if any(dt.kind not in 'biufc' for dt in value.dtypes): # int, bool, int, float, complex
                logging.info(f"Converting DataFrame '{key}' to numpy array with dtype=object.")
                return value.values
            else:
                logging.info(f"Converting numeric DataFrame '{key}' to numpy array with dtype=float32.")
                return value.values.astype(np.float32)

        elif isinstance(value, pd.Series):
            # A Series has a single dtype, so check it directly
            if value.dtype.kind not in 'biufc':
                logging.info(f"Converting Series '{key}' to numpy array with dtype=object.")
                return value.values
            else:
                logging.info(f"Converting numeric Series '{key}' to numpy array with dtype=float32.")
                return value.values.astype(np.float32)
        # --- FIX ENDS HERE ---

        # For lists and tuples, let numpy decide the best dtype
        try:
            array = np.array(value)
            logging.info(f"Converted list/tuple '{key}' to numpy array with dtype={array.dtype}.")
            return array
        except Exception as e:
            raise TypeError(f"Could not convert key '{key}' to a numpy array: {e}") from e

    def _ensure_2d(self, value: np.ndarray, key: str) -> np.ndarray:
        """Ensures a numpy array is 2-dimensional."""
        if value.ndim == 1: return value.reshape(-1, 1)
        if value.ndim != 2: raise ValueError(f"Key '{key}' has unsupported dimensions: {value.ndim}.")
        return value