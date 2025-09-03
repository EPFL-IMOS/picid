import logging
from typing import Any
import numpy as np


# Basic logger configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# 1. Base Class for All Data Objects
# =============================================================================
class BaseDataObject(dict):
    """
    A base object that provides attribute-style access and supports
    recursive conversion of nested dictionaries, but protects 'metadata'.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets an item, converting nested dictionaries to BaseDataObjects,
        but leaving the 'metadata' key as a plain dictionary.
        """
        # --- FIX ---
        # Protect the 'metadata' key from being converted into a BaseDataObject.
        if isinstance(value, dict):
            value = BaseDataObject(**value)
        super().__setitem__(key, value)

    # ... (the rest of the class is unchanged)
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from e

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
        rows = [
            f'"{key}":'.ljust(max_key_length + 4) + repr(value)
            for key, value in sorted(self.items())
        ]
        return "\n".join(rows)
