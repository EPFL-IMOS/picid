# In a new file, e.g., data/validation.py
import warnings
from typing import Any, List, Optional
import numpy as np
import pandas as pd

def get_length(value: Any) -> Optional[int]:
    """Helper to safely get the length of different array-like types."""
    if hasattr(value, 'shape') and hasattr(value.shape, '__len__') and value.shape:
        return value.shape[0]
    if isinstance(value, (list, tuple)):
        return len(value)
    return None

def check_length_consistency(
    values: List[Any],
    keys: List[str],
    raise_on_error: bool = True
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
    values: List[Any],
    keys: List[str],
    raise_on_error: bool = True
) -> None:
    """Checks for NaN values in pandas or numpy objects."""
    for value, key in zip(values, keys):
        has_nan = False
        if isinstance(value, (pd.DataFrame, pd.Series)):
            numeric_data = value.select_dtypes(include=np.number)
            if not numeric_data.empty:
                has_nan = numeric_data.isnull().values.any()
        elif isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.number):
                has_nan = np.isnan(value).any()
        
        if has_nan:
            message = f"NaN values found in object '{key}'."
            if raise_on_error:
                raise ValueError(message)
            else:
                warnings.warn(message)