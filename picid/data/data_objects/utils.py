import warnings
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import awkward as ak


def get_length(value: Any) -> Optional[int]:
    """
    Return the leading dimension length for a value.

    Parameters
    ----------
    value : Any
        Array-like object to inspect.

    Returns
    -------
    int | None
        Leading dimension length, or ``None`` when the value has no length.
    """
    if hasattr(value, "shape") and hasattr(value.shape, "__len__") and value.shape:
        return value.shape[0]
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def check_length_consistency(
    values: List[Any], keys: List[str], raise_on_error: bool = True
) -> None:
    """
    Check that all provided array-like items have the same length.

    Parameters
    ----------
    values : list[Any]
        Array-like values to compare.
    keys : list[str]
        Names used in the error message.
    raise_on_error : bool, default=True
        Whether to raise an exception instead of emitting a warning.
    """
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
    """
    Check for NaN values in common array-like containers.

    Parameters
    ----------
    values : list[Any]
        Data objects to inspect.
    keys : list[str]
        Names corresponding to the data objects.
    raise_on_error : bool, default=True
        Whether to raise an exception instead of emitting a warning.

    Raises
    ------
    ValueError
        If NaNs are found and raise_on_error is True.
    """
    for value, key in zip(values, keys):
        has_nan = False

        # 1. Handle pandas DataFrames
        if isinstance(value, pd.DataFrame):
            # Check only numeric columns for NaNs
            numeric_data = value.select_dtypes(include=np.number)
            if not numeric_data.empty:
                has_nan = numeric_data.isnull().values.any()

        # 2. Handle pandas Series
        elif isinstance(value, pd.Series):
            # Check Series only if it's a numeric type
            if pd.api.types.is_numeric_dtype(value.dtype):
                has_nan = value.isnull().any()

        # 3. Handle numpy arrays
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                has_nan = np.isnan(value).any()

        # 4. Handle awkward arrays (numeric only; non-numeric e.g. string raises in np.isnan)
        elif isinstance(value, ak.Array):
            try:
                # Optimized check: ak.any(np.isnan()) is significantly faster
                # than converting to numpy or flattening for standard ragged arrays.
                # It operates directly on the structure.
                has_nan = bool(ak.any(np.isnan(value)))
            except TypeError:
                # Non-numeric content (e.g. string type): skip NaN check
                pass

        # Reporting
        if has_nan:
            message = f"NaN values found in object '{key}'."
            if raise_on_error:
                raise ValueError(message)
            else:
                warnings.warn(message)


def convert_to_numpy(
    value: Any, ensure_2d: bool = True
) -> Union[np.ndarray, pd.Series]:
    """
    Convert array-like data into a NumPy array.

    Parameters
    ----------
    value : Any
        Value to convert.
    ensure_2d : bool, default=True
        Whether to reshape 1D arrays into a column vector.

    Returns
    -------
    numpy.ndarray | pandas.Series
        Converted array, or the original Series when non-numeric.
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
