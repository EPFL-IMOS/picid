"""Return wrappers used by the transform and preprocessing pipeline."""

from __future__ import annotations

import warnings
from typing import Generic, TypeVar

import awkward as ak
import numpy as np

from picid.data.data_objects.core.metadata_data_object import BaseDataObjectWithMetadata
from picid.data.data_objects.utils import check_for_nans, check_length_consistency

T = TypeVar("T", bound=(np.ndarray | ak.Array))


class NamedTransformInput(BaseDataObjectWithMetadata):
    """Store transform inputs and provide optional validation before use."""

    def get_sanitized_data(
        self,
        check_nans: bool = True,
        check_lengths: bool = True,
        warn_only: bool = False,
        size_threshold: int = 1_000_000,
        return_copy: bool = False,
    ) -> "NamedTransformInput":
        """
        Perform validation, extract column metadata, and convert data to NumPy arrays.

        Parameters
        ----------
        check_nans : bool, default=True
            Whether to check the stored payloads for NaN values.
        check_lengths : bool, default=True
            Whether to verify equal first-dimension lengths across payloads.
        warn_only : bool, default=False
            Whether to warn instead of raising on validation failures.
        size_threshold : int, default=1_000_000
            Maximum total size below which length consistency is checked.
        return_copy : bool, default=False
            Whether to validate a copy instead of mutating the current object.

        Returns
        -------
        NamedTransformInput
            The validated object, or a validated copy when requested.
        """
        target_obj = self.copy() if return_copy else self

        values = list(target_obj.values())
        keys = list(target_obj.keys())

        if check_nans:
            check_for_nans(values, keys, raise_on_error=(not warn_only))

        if check_lengths:
            total_size = sum(v.size for v in values if hasattr(v, "size"))
            if total_size < size_threshold:
                check_length_consistency(values, keys, raise_on_error=(not warn_only))
            else:
                warnings.warn(
                    f"Data size ({total_size}) exceeds threshold ({size_threshold}). "
                    "Skipping length consistency check for performance."
                )

        return target_obj


class ReturnObject(BaseDataObjectWithMetadata, Generic[T]):
    """A generic return object for transformations."""

    pass


class SimpleReturnObject(ReturnObject, Generic[T]):
    """A data object for simple, single-array returns from a transform."""

    pass


class NamedDictReturnObject(ReturnObject, Generic[T]):
    """A data object for dictionary-like returns with named keys."""

    pass


class ExtendedReturnObject(ReturnObject, Generic[T]):
    """A data object for complex returns that may include data and metadata."""

    pass
