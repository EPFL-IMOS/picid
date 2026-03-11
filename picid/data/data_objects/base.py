import copy
import logging
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from collections.abc import MutableMapping

import numpy as np
import pandas as pd
import awkward as ak


from picid.data.data_objects.utils import (
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
    BaseDataObject is a generic, dict-like container for storing and managing data objects of a specified type.
    This class provides both dictionary-style and attribute-style access to its items, and supports mutable mapping operations.
    It can recursively construct nested BaseDataObject instances from dictionaries, and validates types for list values.
    Parameters
    ----------
    **kwargs : Any
        Initial key-value pairs to populate the object.
    Attributes
    ----------
    _instance_cls : Dict[str, Type[T]]
        The class type of the data objects.
    _data : Dict[str, T]
        Internal storage for the data objects.
    Methods
    -------
    __getitem__(key: str) -> T
        Retrieve the value associated with the given key.
    __setitem__(key: str, value: Any) -> None
        Set the value for the given key, with type validation and recursive construction for dicts.
    __delitem__(key: str) -> None
        Remove the item associated with the given key.
    __contains__(key: str) -> bool
        Check if the key exists in the object.
    __len__() -> int
        Return the number of items.
    __iter__()
        Return an iterator over the keys.
    items()
        Return a view of the object's items.
    keys()
        Return a view of the object's keys.
    values()
        Return a view of the object's values.
    get_instance_cls() -> Dict[str, Type[T]]
        Return the class type of the stored data objects.
    __getattr__(key: str) -> T
        Provide attribute-style access to items (read-only).
    __repr__() -> str
        Return a string representation of the object.
    copy(deep: bool = True) -> "BaseDataObject[T]"
        Return a copy of the object, optionally deep copying its contents.
    Notes
    -----
    - When setting a value as a dict, it is recursively converted into a BaseDataObject.
    - When setting a value as a list, all items must be of the specified instance class type.
    - Attribute-style access is read-only and raises AttributeError for missing keys.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._instance_cls: Dict[str, Type[T]] = {}
        self._data: Dict[str, T] = {}
        for key, value in kwargs.items():
            self[key] = value  # go through __setitem__

    # --- dict-like interface ---
    def __getitem__(self, key: str) -> T:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            value = self.__class__(**value)

        # Enforce allowed types: pandas (DataFrame/Series), numpy (ndarray), or awkward (Array).
        allowed_single_types = (pd.DataFrame, pd.Series, np.ndarray, ak.Array)

        if isinstance(value, list):
            if len(value) > 0:
                type_set = {type(v) for v in value}
                if len(type_set) != 1:
                    logger.debug(
                        f"All items in the list must have the same type, got {type_set}"
                    )
                elem_type = next(iter(type_set))
                if not issubclass(elem_type, allowed_single_types):
                    logger.debug(
                        f"List items must be pandas (DataFrame/Series), "
                        f"numpy (ndarray), or awkward (Array), got {elem_type.__name__}"
                    )

                if key not in self._instance_cls:
                    self._instance_cls[key] = elem_type
                else:
                    if self._instance_cls[key] != elem_type:
                        logger.debug(
                            f"Expected list items of type {self._instance_cls[key].__name__}, "
                            f"got {elem_type.__name__}"
                        )
                self._data[key] = value
            else:
                logger.debug("Received an empty list; type cannot be validated.")

        elif isinstance(value, allowed_single_types):
            if key not in self._instance_cls:
                self._instance_cls[key] = type(value)
            else:
                if self._instance_cls[key] != type(value):
                    logger.debug(
                        f"Expected type {self._instance_cls[key].__name__}, got {type(value).__name__}. Changing stored type."
                    )
                    self._instance_cls[key] = type(value)
            self._data[key] = value
        elif isinstance(value, BaseDataObject):
            self._instance_cls[key] = type(value)
            self._data[key] = value
        else:
            raise TypeError(
                f"Value must be a dict, list of allowed types, or single allowed type. "
                f"Allowed types: {', '.join(t.__name__ for t in allowed_single_types)}. "
                f"Got {type(value).__name__} for key '{key}'."
            )

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

    def get_instance_cls(self) -> Dict[str, Type[T]]:
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
        new = cls()
        if deep:
            # recursively deep copy the dict
            new._instance_cls = copy.deepcopy(self._instance_cls)
            new._data = copy.deepcopy(self._data)
        else:
            # shallow copy of the dict
            new._instance_cls = self._instance_cls.copy()
            new._data = self._data.copy()
        return new


class BaseDataObjectWithMetadata(
    BaseDataObject[T], ToNumpyMixin, ToDataFrameMixin, Generic[T]
):
    """
    A flexible data container that stores an individual data unit in its
    original format.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Initializes the chunk, handling metadata and column extraction robustly.
        """
        # We keep Metadata as a private attribute to keep the data homogeneous
        self._metadata: Optional[Dict[str, Any]] = None

        # Initialize all other data first
        super().__init__(**kwargs)

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
