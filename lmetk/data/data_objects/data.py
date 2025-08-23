import logging
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from lmetk.data.data_objects.base import BaseDataObject # Assuming your BaseDataObject is here

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DataChunk(BaseDataObject):
    """
    A flexible data container that stores data in its original format (Lazy Conversion).

    Data is stored as-is (e.g., pandas.DataFrame). Conversion to NumPy is handled
    explicitly via the .to_numpy() method. Validation is also an explicit step.
    """
    # --- Public API Methods ---

    def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
        """
        Converts the data for a given key to a NumPy array on demand.

        This is the primary method for getting a NumPy representation of your data.

        Args:
            key (str): The key of the item to convert.
            ensure_2d (bool): If True, ensures the output array is 2-dimensional.

        Returns:
            np.ndarray: The converted NumPy array.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DatasetChunk.")

        value = self[key]

        # --- Conversion Logic ---
        if isinstance(value, np.ndarray):
            array = value
        elif isinstance(value, (pd.DataFrame, pd.Series)):
            if any(dt.kind not in 'biufc' for dt in getattr(value, 'dtypes', [getattr(value, 'dtype', None)])):
                array = value.values
            else:
                array = value.values.astype(np.float32)
        else: # list, tuple
            array = np.array(value)

        # --- Dimension check ---
        if ensure_2d:
            if array.ndim == 1:
                return array.reshape(-1, 1)
            if array.ndim != 2:
                raise ValueError(f"Key '{key}' has unsupported dimensions: {array.ndim}.")
        return array

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame representation of the data for a given key.

        If the item is already a DataFrame, it's returned directly.
        Otherwise, it's converted.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DatasetChunk.")
        
        value = self[key]
        if isinstance(value, pd.DataFrame):
            return value # Return original object if it's already a DataFrame

        # Otherwise, convert to DataFrame
        array_data = self.to_numpy(key, ensure_2d=False) # Get numpy version
        columns = self.get("metadata", {}).get("column_map", {}).get(key)
        return pd.DataFrame(array_data, columns=columns)

    def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
        """
        Performs length and NaN consistency checks on the raw data.
        """
        keys = keys_to_validate if keys_to_validate is not None else self.keys()
        items_to_check = {k: self[k] for k in keys if k in self}
        
        if not items_to_check:
            logging.info("No items to validate.")
            return

        self._check_for_nans(list(items_to_check.values()), list(items_to_check.keys()))
        self._check_length_consistency(list(items_to_check.values()), list(items_to_check.keys()))
        logging.info("Validation successful for keys: %s", list(items_to_check.keys()))

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

        super().__init__(**kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        """Sets an item by storing it in its original format."""
        logging.info(f"Storing key '{key}' as-is (type: {type(value).__name__}).")
        super().__setitem__(key, value)

    # --- Internal Validation Helpers (adapted for mixed types) ---

    def _get_length(self, value: Any) -> Optional[int]:
        """Helper to get the length of different array-like types."""
        if hasattr(value, 'shape') and value.shape: # Pandas, Numpy
            return value.shape[0]
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    def _check_length_consistency(self, values: List[Any], keys: List[str]) -> None:
        """Checks that all provided items have the same length."""
        lengths = {self._get_length(v) for v in values if self._get_length(v) is not None}
        if len(lengths) > 1:
            length_info = {k: self._get_length(v) for k, v in zip(keys, values)}
            raise ValueError(f"Length mismatch between items: {length_info}")

    def _check_for_nans(self, values: List[Any], keys: List[str]) -> None:
        """Checks for NaNs in various raw data types."""
        for value, key in zip(values, keys):
            has_nan = False
            if isinstance(value, (pd.DataFrame, pd.Series)):
                numeric_data = value.select_dtypes(include=np.number)
                if not numeric_data.empty: has_nan = numeric_data.isnull().values.any()
            elif isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.number): has_nan = np.isnan(value).any()
            if has_nan:
                raise ValueError(f"NaN values found in object '{key}'.")


# import logging
# from typing import Any, List, Optional
# import numpy as np
# import pandas as pd
# from lmetk.data.data_objects.base import BaseDataObject

# # Basic logger configuration
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# class DatasetChunk(BaseDataObject):
#     """
#     A specialized data container for holding and processing array-like PHM data.

#     It automatically converts pandas objects and lists into NumPy arrays while
#     preserving metadata and tracking original DataFrame columns. Length consistency
#     must be checked explicitly by calling the .validate() method.
#     """
#     ARRAY_LIKE_TYPES = (pd.DataFrame, pd.Series, np.ndarray, list, tuple)

#     # --- Public API Methods ---

#     def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
#         """
#         Performs length and NaN consistency checks on specified numpy arrays.

#         If no keys are specified, all numpy arrays in the chunk will be validated.
#         This is the explicit way to check for length mismatches.

#         Args:
#             keys_to_validate (Optional[List[str]]): A list of keys to validate.
#         """
#         all_array_items = {k: v for k, v in self.items() if isinstance(v, np.ndarray)}
#         items_to_check = all_array_items if keys_to_validate is None else {
#             k: all_array_items[k] for k in keys_to_validate if k in all_array_items
#         }

#         if not items_to_check:
#             logging.info("No arrays to validate.")
#             return

#         keys = list(items_to_check.keys())
#         arrays = list(items_to_check.values())

#         # Perform all validations here
#         self._check_length_consistency(arrays, keys)
#         self._check_for_nans(arrays, keys) # Note: _check_for_nans on arrays is faster
#         logging.info("Full validation successful for keys: %s", keys)

#     def to_dataframe(self, key: str) -> pd.DataFrame:
#         """
#         Converts a specified numpy array back to a pandas DataFrame.
#         """
#         if key not in self or not isinstance(self[key], np.ndarray):
#             raise AttributeError(f"Cannot create DataFrame: key '{key}' with a numpy array not found.")
#         array_data, columns = self[key], self.get("metadata", {}).get("column_map", {}).get(key)
#         if columns and len(columns) != array_data.shape[1]:
#             logging.warning(f"Column count mismatch for '{key}': Metadata has {len(columns)} names, but array has {array_data.shape[1]} columns. Ignoring names.")
#             columns = None
#         return pd.DataFrame(array_data, columns=columns)

#     # --- Dunder Methods ---

#     def __init__(self, **kwargs: Any) -> None:
#         """Initializes the chunk and handles DataFrame column extraction."""
#         # Auto-extract column names from ANY DataFrame into a nested metadata map
#         if any(isinstance(v, pd.DataFrame) for v in kwargs.values()):
#             kwargs.setdefault('metadata', {})
#             kwargs['metadata'].setdefault('column_map', {})
#         for key, value in kwargs.items():
#             if isinstance(value, pd.DataFrame):
#                 logging.info(f"Extracting columns for DataFrame '{key}' into metadata.column_map.")
#                 kwargs['metadata']['column_map'][key] = value.columns.tolist()
        
#         # Call the parent's __init__ to handle the actual setting of items
#         super().__init__(**kwargs)

#     def __setitem__(self, key: str, value: Any) -> None:
#         """Sets an item, converting array-likes and preserving metadata."""
#         if isinstance(value, self.ARRAY_LIKE_TYPES):
#             # Perform pre-conversion NaN check on the raw object
#             self._check_for_nans_pre_conversion([value], [key])
#             # Process the value
#             processed_value = self._convert_to_numpy(value, key)
#             processed_value = self._ensure_2d(processed_value, key)
#             super().__setitem__(key, processed_value)
#         else:
#             # For non-array types, just store them
#             logging.info(f"Storing key '{key}' as-is (type: {type(value).__name__}).")
#             super().__setitem__(key, value)

#     # --- Internal Validation Helpers ---

#     def _check_length_consistency(self, arrays: List[np.ndarray], keys: List[str]) -> None:
#         """Checks that all provided numpy arrays have the same number of rows."""
#         if not arrays: return
#         lengths = {arr.shape[0] for arr in arrays}
#         if len(lengths) > 1:
#             length_info = {k: v.shape[0] for k, v in zip(keys, arrays)}
#             raise ValueError(f"Length mismatch between arrays: {length_info}")

#     def _check_for_nans(self, arrays: List[np.ndarray], keys: List[str]) -> None:
#         """Checks for NaNs in numpy arrays (more efficient for validation)."""
#         for arr, key in zip(arrays, keys):
#             if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
#                 raise ValueError(f"NaN values found in array '{key}'.")

#     def _check_for_nans_pre_conversion(self, values: List[Any], keys: List[str]) -> None:
#         """Checks for NaNs in various raw data types before conversion."""
#         for value, key in zip(values, keys):
#             has_nan = False
#             # Handle pandas DataFrames
#             if isinstance(value, pd.DataFrame):
#                 numeric_data = value.select_dtypes(include=np.number)
#                 if not numeric_data.empty:
#                     has_nan = numeric_data.isnull().values.any()

#             # --- FIX STARTS HERE ---
#             # Handle pandas Series separately
#             elif isinstance(value, pd.Series):
#                 # A Series has one dtype, so check if it's numeric before looking for NaNs
#                 if pd.api.types.is_numeric_dtype(value.dtype):
#                     has_nan = value.isnull().any()
#             # --- FIX ENDS HERE ---

#             # Handle numpy arrays
#             elif isinstance(value, np.ndarray):
#                 if np.issubdtype(value.dtype, np.number):
#                     has_nan = np.isnan(value).any()

#             # Handle lists/tuples
#             elif isinstance(value, (list, tuple)):
#                 try:
#                     numeric_list = [x for x in value if isinstance(x, (int, float))]
#                     if numeric_list and np.isnan(np.array(numeric_list)).any():
#                         has_nan = True
#                 except TypeError:
#                     pass
            
#             if has_nan:
#                 raise ValueError(f"NaN values found in object '{key}' before conversion.")

#     # --- Internal Processing Helpers ---

#     def _convert_to_numpy(self, value: Any, key: str) -> np.ndarray:
#         """
#         Converts array-like objects to a numpy array with an appropriate dtype.
#         """
#         if isinstance(value, np.ndarray):
#             return value

#         # --- FIX STARTS HERE ---
#         # For pandas objects, inspect dtypes to decide conversion strategy
#         if isinstance(value, pd.DataFrame):
#             # If there are any non-numeric types, use 'object' dtype for safety
#             if any(dt.kind not in 'biufc' for dt in value.dtypes): # int, bool, int, float, complex
#                 logging.info(f"Converting DataFrame '{key}' to numpy array with dtype=object.")
#                 return value.values
#             else:
#                 logging.info(f"Converting numeric DataFrame '{key}' to numpy array with dtype=float32.")
#                 return value.values.astype(np.float32)

#         elif isinstance(value, pd.Series):
#             # A Series has a single dtype, so check it directly
#             if value.dtype.kind not in 'biufc':
#                 logging.info(f"Converting Series '{key}' to numpy array with dtype=object.")
#                 return value.values
#             else:
#                 logging.info(f"Converting numeric Series '{key}' to numpy array with dtype=float32.")
#                 return value.values.astype(np.float32)
#         # --- FIX ENDS HERE ---

#         # For lists and tuples, let numpy decide the best dtype
#         try:
#             array = np.array(value)
#             logging.info(f"Converted list/tuple '{key}' to numpy array with dtype={array.dtype}.")
#             return array
#         except Exception as e:
#             raise TypeError(f"Could not convert key '{key}' to a numpy array: {e}") from e

#     def _ensure_2d(self, value: np.ndarray, key: str) -> np.ndarray:
#         """Ensures a numpy array is 2-dimensional."""
#         if value.ndim == 1: return value.reshape(-1, 1)
#         if value.ndim != 2: raise ValueError(f"Key '{key}' has unsupported dimensions: {value.ndim}.")
#         return value