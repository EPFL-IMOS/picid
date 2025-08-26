import numpy as np
import pandas as pd
import torch
from typing import Any, Tuple, Type

# --- Utility for Type Conversion ---
def _convert_to_numpy(data: Any) -> Tuple[np.ndarray, Type]:
    """Converts input data to numpy.ndarray and returns its original type."""
    original_type = type(data)
    
    if isinstance(data, np.ndarray):
        return data, original_type
    elif isinstance(data, pd.DataFrame):
        return data.values, original_type
    elif isinstance(data, pd.Series):
        return data.values.reshape(-1,1), original_type
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy(), original_type # Move to CPU if on GPU, then convert
    else:
        # For other types, try direct conversion or raise error if unsupported
        try:
            return np.asarray(data), original_type
        except Exception as e:
            raise TypeError(f"Unsupported data type for conversion to numpy: {original_type}. Error: {e}")

def _convert_from_numpy(numpy_data: np.ndarray, original_type: Type, original_data: Any) -> Any:
    """Converts numpy.ndarray back to the original data type."""
    if original_type is np.ndarray:
        return numpy_data
    elif original_type is pd.DataFrame:
        # Attempt to reconstruct DataFrame, preserving index and columns if possible
        if isinstance(original_data, pd.DataFrame):
            return pd.DataFrame(numpy_data, index=original_data.index, columns=original_data.columns)
        return pd.DataFrame(numpy_data) # Fallback
    elif original_type is pd.Series:
        # Attempt to reconstruct Series, preserving index and name if possible
        if isinstance(original_data, pd.Series):
            return pd.Series(numpy_data.flatten(), index=original_data.index, name=original_data.name)
        return pd.Series(numpy_data.flatten()) # Fallback
    elif original_type is torch.Tensor:
        # Convert back to Tensor, preserving original device if possible
        if isinstance(original_data, torch.Tensor):
            return torch.from_numpy(numpy_data).to(original_data.device)
        return torch.from_numpy(numpy_data) # Fallback to CPU
    else:
        # If the original type was not explicitly handled, return numpy array
        # as it was the intermediate format. This might lead to unexpected types
        # downstream if not careful.
        return numpy_data
