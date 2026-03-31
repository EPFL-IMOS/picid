"""
Core dict-like data-object primitive used across the data pipeline.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import MutableMapping
from typing import Any, Generic, Type, TypeVar

import awkward as ak
import numpy as np
import pandas as pd

T = TypeVar("T")
logger = logging.getLogger(__name__)


class BaseDataObject(Generic[T], MutableMapping[str, T]):
    """
    Store typed payloads with both mapping-style and attribute-style access.

    Parameters
    ----------
    **kwargs : Any
        Initial key-value pairs to store in the object.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the object with any provided key-value pairs.

        Parameters
        ----------
        **kwargs : Any
            Initial key-value pairs to store.
        """
        self._instance_cls: dict[str, Type[T]] = {}
        self._data: dict[str, T] = {}
        for key, value in kwargs.items():
            self[key] = value

    def __getitem__(self, key: str) -> T:
        """
        Return the value stored under ``key``.

        Parameters
        ----------
        key : str
            Stored key to retrieve.

        Returns
        -------
        T
            Value associated with ``key``.
        """
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Store ``value`` under ``key`` after type normalization and checks.

        Parameters
        ----------
        key : str
            Key to store.
        value : Any
            Value to normalize and store.
        """
        if isinstance(value, dict):
            value = self.__class__(**value)

        allowed_single_types = (pd.DataFrame, pd.Series, np.ndarray, ak.Array)

        if isinstance(value, list):
            if len(value) > 0:
                type_set = {type(v) for v in value}
                if len(type_set) != 1:
                    logger.debug(
                        "All items in the list must have the same type, got %s",
                        type_set,
                    )
                elem_type = next(iter(type_set))
                if not issubclass(elem_type, allowed_single_types):
                    logger.debug(
                        "List items must be pandas (DataFrame/Series), numpy (ndarray), "
                        "or awkward (Array), got %s",
                        elem_type.__name__,
                    )

                if key not in self._instance_cls:
                    self._instance_cls[key] = elem_type
                elif self._instance_cls[key] != elem_type:
                    logger.debug(
                        "Expected list items of type %s, got %s",
                        self._instance_cls[key].__name__,
                        elem_type.__name__,
                    )
                self._data[key] = value
            else:
                logger.debug(
                    "Received an empty list for key '%s'; preserving it without "
                    "recording an element type.",
                    key,
                )
                self._data[key] = value

        elif isinstance(value, allowed_single_types):
            if key not in self._instance_cls:
                self._instance_cls[key] = type(value)
            elif not isinstance(value, self._instance_cls[key]):
                logger.debug(
                    "Expected type %s, got %s. Changing stored type.",
                    self._instance_cls[key].__name__,
                    type(value).__name__,
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
        """
        Remove the value stored under ``key``.

        Parameters
        ----------
        key : str
            Stored key to remove.
        """
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """
        Return whether ``key`` exists in the object.

        Parameters
        ----------
        key : str
            Key to check.

        Returns
        -------
        bool
            ``True`` when the key is present.
        """
        return key in self._data

    def __len__(self) -> int:
        """
        Return the number of stored keys.

        Returns
        -------
        int
            Number of keys currently stored.
        """
        return len(self._data)

    def __iter__(self):
        """
        Iterate over the stored keys.

        Returns
        -------
        collections.abc.Iterator[str]
            Iterator over the stored keys.
        """
        return iter(self._data)

    def items(self):
        """
        Return the stored key-value items.

        Returns
        -------
        dict_items
            Items view over the stored mapping.
        """
        return self._data.items()

    def keys(self):
        """
        Return the stored keys.

        Returns
        -------
        dict_keys
            Keys view over the stored mapping.
        """
        return self._data.keys()

    def values(self):
        """
        Return the stored values.

        Returns
        -------
        dict_values
            Values view over the stored mapping.
        """
        return self._data.values()

    def get_instance_cls(self) -> dict[str, Type[T]]:
        """
        Return the inferred value type stored for each key.

        Returns
        -------
        dict[str, type[T]]
            Cached type information for each stored key.
        """
        return self._instance_cls

    def __getattr__(self, key: str) -> T:
        """
        Return ``key`` through attribute-style access when present.

        Parameters
        ----------
        key : str
            Attribute name to resolve.

        Returns
        -------
        T
            Stored value for the requested key.
        """
        data = object.__getattribute__(self, "_data")
        try:
            return data[key]
        except KeyError as error:
            raise AttributeError(
                f"{self.__class__.__name__!r} has no attribute {key!r}"
            ) from error

    def __repr__(self) -> str:
        if not self._data:
            return f"{self.__class__.__name__}()"
        parts = [f"{k}={v!r}" for k, v in self._data.items()]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def copy(self: "BaseDataObject[T]", deep: bool = True) -> "BaseDataObject[T]":
        """
        Return a shallow or deep copy of the object.

        Parameters
        ----------
        deep : bool, default=True
            Whether to deep-copy the stored payloads and type map.

        Returns
        -------
        BaseDataObject[T]
            Copied data object.
        """
        cls: Type[BaseDataObject[T]] = self.__class__
        new = cls()
        if deep:
            new._instance_cls = copy.deepcopy(self._instance_cls)
            new._data = copy.deepcopy(self._data)
        else:
            new._instance_cls = self._instance_cls.copy()
            new._data = self._data.copy()
        return new
