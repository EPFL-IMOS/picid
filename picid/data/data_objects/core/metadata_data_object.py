"""Metadata-aware data-object primitive shared by containers and returns."""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional, TypeVar

import pandas as pd

from picid.data.data_objects.core.base_data_object import BaseDataObject
from picid.data.data_objects.mixins import ToDataFrameMixin, ToNumpyMixin
from picid.data.data_objects.utils import check_for_nans, check_length_consistency

T = TypeVar("T")
logger = logging.getLogger(__name__)


class BaseDataObjectWithMetadata(BaseDataObject[T], ToNumpyMixin, ToDataFrameMixin):
    """
    Store payloads together with optional metadata such as column mappings.

    Parameters
    ----------
    metadata : dict[str, Any] | None, default=None
        Optional metadata dictionary stored alongside the payloads.
    container_metadata : dict[str, Any] | None, default=None
        Canonical alias for ``metadata``.
    **kwargs : Any
        Initial payload key-value pairs.
    """

    def __init__(
        self,
        metadata: Optional[dict[str, Any]] = None,
        container_metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the object, handling metadata and column extraction robustly.

        Parameters
        ----------
        metadata : dict[str, Any] | None, default=None
            Optional metadata dictionary stored alongside the payloads.
        container_metadata : dict[str, Any] | None, default=None
            Canonical alias for ``metadata``.
        **kwargs : Any
            Initial payload key-value pairs.
        """
        if metadata is not None and container_metadata is not None:
            raise ValueError(
                "Pass either 'metadata' or 'container_metadata', not both."
            )

        metadata = container_metadata if container_metadata is not None else metadata

        self._metadata: Optional[dict[str, Any]] = None

        super().__init__(**kwargs)

        if metadata is not None:
            metadata.setdefault("column_map", {})
            for key, value in self.items():
                if isinstance(value, pd.DataFrame):
                    metadata["column_map"][key] = value.columns.tolist()
            self._metadata = metadata
        else:
            self._metadata = {}

    def __getitem__(self, key: str) -> T:
        """
        Return payload values and expose metadata under the ``metadata`` key.

        Parameters
        ----------
        key : str
            Payload key or ``"metadata"``.

        Returns
        -------
        T
            Stored payload or metadata view.
        """
        if key == "metadata":
            return self.metadata  # type: ignore[return-value]
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set an item, storing it in its original format and intelligently updating the column_map.

        Parameters
        ----------
        key : str
            Name of the item to store.
        value : Any
            Payload value to store under ``key``.
        """
        super().__setitem__(key, value)

        if self._metadata is not None and isinstance(self._metadata, dict):
            is_df = isinstance(value, pd.DataFrame)
            self._metadata.setdefault("column_map", {})

            if is_df:
                self._metadata["column_map"][key] = value.columns.tolist()
            elif key in self._metadata.get("column_map", {}):
                del self._metadata["column_map"][key]

    def copy(
        self: "BaseDataObjectWithMetadata[T]", deep: bool = True
    ) -> "BaseDataObjectWithMetadata[T]":
        """
        Return a shallow or deep copy, including metadata.

        Parameters
        ----------
        deep : bool, default=True
            Whether to deep-copy the payload and metadata dictionaries.

        Returns
        -------
        BaseDataObjectWithMetadata[T]
            Copied data object.
        """
        new = super().copy(deep=deep)
        if self._metadata is None:
            new._metadata = None
        else:
            new._metadata = (
                copy.deepcopy(self._metadata) if deep else self._metadata.copy()
            )
        return new

    @property
    def metadata(self) -> Optional[dict[str, Any]]:
        """
        Return the metadata dictionary, if it exists.

        Returns
        -------
        dict[str, Any] | None
            Stored metadata dictionary, if present.
        """
        return self._metadata

    @property
    def container_metadata(self) -> Optional[dict[str, Any]]:
        """
        Compatibility alias for object-level metadata storage.

        Returns
        -------
        dict[str, Any] | None
            Stored metadata dictionary, if present.
        """
        return self.metadata

    def validate(self, keys_to_validate: Optional[list[str]] = None) -> None:
        """
        Perform length and NaN consistency checks using external utilities.

        Parameters
        ----------
        keys_to_validate : list[str] | None, default=None
            Optional subset of keys to validate.
        """
        keys = keys_to_validate if keys_to_validate is not None else self.keys()
        items_to_check = {k: self[k] for k in keys if k in self}

        if not items_to_check:
            logging.info("No items to validate.")
            return

        values = list(items_to_check.values())
        keys = list(items_to_check.keys())

        check_for_nans(values, keys)
        check_length_consistency(values, keys)

        logging.info("Validation successful for keys: %s", keys)
