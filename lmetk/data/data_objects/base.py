import logging
from typing import Any
import numpy as np


# Basic logger configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# 1. Base Class for All Data Objects
# =============================================================================
class BaseDataObject(dict):
    """
    A base object for data containers in the framework.

    It provides dictionary-like storage with convenient attribute-style access
    and a structured initialization process.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        for key, value in kwargs.items():
            self[key] = value

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
        if not self: return ""
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
