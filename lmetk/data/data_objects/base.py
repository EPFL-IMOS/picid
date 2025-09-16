import copy
import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from collections.abc import MutableMapping

import numpy as np
import pandas as pd

from lmetk.data.data_objects import data
from lmetk.data.data_objects.utils import (
    check_for_nans,
    check_length_consistency,
    convert_to_numpy,
)

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ToNumpyMixin:

    def to_numpy(self, key: str, ensure_2d: bool = True) -> np.ndarray:
        """
        Converts the data for a given key to a NumPy array on demand.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in OrderedInput.")
        value = self[key]
        # This calls the standalone utility function we created earlier
        return convert_to_numpy(value, ensure_2d)


class ToDataFrameMixin:

    def to_dataframe(self, key: str) -> pd.DataFrame:
        """
        Returns a pandas.DataFrame representation of the data for a given key.
        """
        if key not in self:
            raise KeyError(f"Key '{key}' not found in DataChunk.")

        value = self[key]
        if isinstance(value, pd.DataFrame):
            return value

        array_data = self.to_numpy(key, ensure_2d=False)
        columns = getattr(self, "metadata", {}).get("column_map", {}).get(key)
        return pd.DataFrame(array_data, columns=columns)


class BaseDataObject(Generic[T], MutableMapping[str, T]):
    """
    A lean base class for managing key–value data with both dict-style and attribute-style access.
    """

    def __init__(self, instance_cls: Type[T], **kwargs: Any) -> None:
        self._instance_cls: Type[T] = instance_cls
        self._data: Dict[str, T] = {}
        for key, value in kwargs.items():
            self[key] = value  # go through __setitem__

    # --- dict-like interface ---
    def __getitem__(self, key: str) -> T:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            value = self.__class__(instance_cls=self._instance_cls, **value)
        if isinstance(value, list):
            type_set = set([type(data) for data in value])
            if len(type_set) != 1 or not all(
                issubclass(t, self._instance_cls) for t in type_set
            ):
                logger.warning(
                    f"All items in the list must be of type {self._instance_cls.__name__}, got {type_set}"
                )
        elif not isinstance(value, (self._instance_cls, self.__class__)):
            logger.warning(
                f"Expected dict, {self._instance_cls.__name__}, "
                f"or {self.__class__.__name__}, got {type(value).__name__}"
            )
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get_instance_cls(self) -> Type[T]:
        return self._instance_cls

    # --- attribute-style access (read-only for convenience) ---
    def __getattr__(self, key: str) -> T:
        data = object.__getattribute__(self, "_data")
        try:
            return data[key]
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {key!r}"
            ) from e

    # --- repr ---
    def __repr__(self) -> str:
        if not self._data:
            return f"{self.__class__.__name__}()"
        parts = [f"{k}={v!r}" for k, v in self._data.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def copy(self: "BaseDataObject[T]", deep: bool = True) -> "BaseDataObject[T]":
        cls: Type[BaseDataObject[T]] = self.__class__
        new = cls(instance_cls=self._instance_cls)
        if deep:
            # recursively deep copy the dict
            new._data = copy.deepcopy(self._data)
        else:
            # shallow copy of the dict
            new._data = self._data.copy()
        return new


class BaseDataObjectWithMetadata(
    BaseDataObject[T], ToNumpyMixin, ToDataFrameMixin, Generic[T]
):
    """
    A flexible data container that stores an individual data unit in its
    original format.
    """

    def __init__(self, instance_cls: Type[T], **kwargs: Any) -> None:
        """
        Initializes the chunk, handling metadata and column extraction robustly.
        """
        # --- FIX ---
        # Pop metadata to handle it separately after other items are initialized.
        # This avoids order-of-operation bugs.

        # We keep Metadata as a private attribute to
        # keep the data homogeneous
        self._metadata: Optional[Dict[str, Any]] = None

        metadata = kwargs.pop("metadata", None)

        # Initialize all other data first
        super().__init__(instance_cls=instance_cls, **kwargs)

        # Now, if metadata was provided, prepare it and set it.
        if metadata is not None:
            # Ensure column_map exists in the user's provided metadata
            metadata.setdefault("column_map", {})
            # Now that `self` is populated, find all DataFrames and update the map
            for key, value in self.items():
                if isinstance(value, pd.DataFrame):
                    metadata["column_map"][key] = value.columns.tolist()

            # Finally, set the fully prepared metadata object
            self._metadata = metadata
        else:
            self._metadata = {}

    # --- dict-like interface ---
    def __getitem__(self, key: str) -> T:
        if key == "metadata":
            return self.metadata  # type: ignore[return-value]
        else:
            return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Sets an item, storing it in its original format and intelligently
        updating the column_map.
        """
        # --- FIX ---
        # Call the parent's (BaseDataObject's) setter first.
        # This handles the recursive logic for non-metadata keys.
        super().__setitem__(key, value)

        # After setting, manage the column map if metadata exists.
        if self._metadata is not None and isinstance(self._metadata, dict):
            is_df = isinstance(value, pd.DataFrame)
            self._metadata.setdefault("column_map", {})

            if is_df:
                self._metadata["column_map"][key] = value.columns.tolist()
            elif key in self._metadata.get("column_map", {}):
                # If a key that had columns is being replaced by a non-DataFrame,
                # remove its stale entry from the map.
                del self._metadata["column_map"][key]

    def copy(
        self: "BaseDataObjectWithMetadata[T]", deep: bool = True
    ) -> "BaseDataObjectWithMetadata[T]":
        new = super().copy(deep=deep)  # this already builds the _data
        # fix up metadata separately
        if self._metadata is None:
            new._metadata = None
        else:
            new._metadata = (
                copy.deepcopy(self._metadata) if deep else self._metadata.copy()
            )
        return new

    # --- Public API Methods ---
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
        """
        Returns the metadata dictionary, if it exists.
        """
        return self._metadata

    def validate(self, keys_to_validate: Optional[List[str]] = None) -> None:
        """
        Performs length and NaN consistency checks using external utilities.
        """
        keys = keys_to_validate if keys_to_validate is not None else self.keys()
        items_to_check = {k: self[k] for k in keys if k in self}

        if not items_to_check:
            logging.info("No items to validate.")
            return

        # Perform validation checks by calling the imported utility functions
        values = list(items_to_check.values())
        keys = list(items_to_check.keys())

        check_for_nans(values, keys)
        check_length_consistency(values, keys)

        logging.info("Validation successful for keys: %s", keys)
