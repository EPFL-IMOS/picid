import copy
import logging
from typing import Any, Dict, Generic, Type, TypeVar
from collections.abc import MutableMapping

import numpy as np

# Basic logger configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

T = TypeVar("T")
logger = logging.getLogger(__name__)


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
