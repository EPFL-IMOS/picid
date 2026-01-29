# In a new file, e.g., data/validation.py
import warnings
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd


def get_length(value: Any) -> Optional[int]:
    """Helper to safely get the length of different array-like types."""
    if hasattr(value, "shape") and hasattr(value.shape, "__len__") and value.shape:
        return value.shape[0]
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def check_length_consistency(
    values: List[Any], keys: List[str], raise_on_error: bool = True
) -> None:
    """Checks that all provided array-like items have the same length."""
    lengths = {get_length(v) for v in values if get_length(v) is not None}
    if len(lengths) > 1:
        length_info = {k: get_length(v) for k, v in zip(keys, values)}
        message = f"Length mismatch between items: {length_info}"
        if raise_on_error:
            raise ValueError(message)
        else:
            warnings.warn(message)


def check_for_nans(
    values: List[Any], keys: List[str], raise_on_error: bool = True
) -> None:
    """Checks for NaN values in various raw data types."""
    for value, key in zip(values, keys):
        has_nan = False

        # Handle pandas DataFrames
        if isinstance(value, pd.DataFrame):
            # Check only numeric columns for NaNs
            numeric_data = value.select_dtypes(include=np.number)
            if not numeric_data.empty:
                has_nan = numeric_data.isnull().values.any()

        # Handle pandas Series separately
        elif isinstance(value, pd.Series):
            # Check Series only if it's a numeric type
            if pd.api.types.is_numeric_dtype(value.dtype):
                has_nan = value.isnull().any()

        # Handle numpy arrays
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                has_nan = np.isnan(value).any()

        if has_nan:
            message = f"NaN values found in object '{key}'."
            if raise_on_error:
                raise ValueError(message)
            else:
                warnings.warn(message)


# Here is another implementation that was found in data.data_objects.udata.py
# If the implementation is preferred, please replace the above with this one.
# Keeping it for debugging purposes.


# def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
#     """
#     Converts the data for a given key to a NumPy array on demand.
#     """
#     if key not in self:
#         raise KeyError(f"Key '{key}' not found in DataChunk.")
#     value = self[key]

#     if isinstance(value, np.ndarray):
#         array = value
#     elif isinstance(value, (pd.DataFrame, pd.Series)):
#         dtypes = getattr(value, "dtypes", [getattr(value, "dtype", None)])
#         if any(dt.kind not in "biufc" for dt in dtypes):
#             array = value.values
#         else:
#             array = value.values.astype(np.float32)
#     else:
#         array = np.array(value)
#     if ensure_2d:
#         if array.ndim == 1:
#             return array.reshape(-1, 1)
#         if array.ndim != 2:
#             raise ValueError(
#                 f"Key '{key}' has unsupported dimensions: {array.ndim}."
#             )
#     return array


def convert_to_numpy(
    value: Any, ensure_2d: bool = True
) -> Union[np.ndarray, pd.Series]:
    """
    Converts array-like data into a NumPy array.

    If a non-numeric pandas Series is provided, it issues a warning and
    returns the original Series unconverted.
    """
    if isinstance(value, np.ndarray):
        array = value

    elif isinstance(value, pd.DataFrame):
        # DataFrames are typically feature matrices; convert them even if it results in an object dtype.
        if any(dt.kind not in "biufc" for dt in value.dtypes):
            array = value.values
        else:
            array = value.values.astype(np.float32)

    elif isinstance(value, pd.Series):
        # If the Series is non-numeric (e.g., datetime, string/object), keep it as a Series.
        if value.dtype.kind not in "biufc":
            warnings.warn(
                f"Has a non-numeric type ({value.dtype}) and will be kept as a pandas Series."
            )
            return value  # Return the original Series and exit the function
        # Otherwise, if it's numeric, convert it.
        else:
            array = value.values.astype(np.float32)

    else:  # For lists, tuples, etc.
        array = np.array(value)

    # This block is now only reached by data that was successfully converted to an array
    if ensure_2d:
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(
                f"Array has unsupported dimensions: {array.ndim}. Set ensure_2d to False to avoid this check."
            )
    return array
