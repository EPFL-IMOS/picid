"""
Canonical dataset containers for split-aware pipeline payloads.
"""

from __future__ import annotations

import logging
import copy
from collections.abc import Mapping
from typing import Any, Generic, Optional, TypeVar

import awkward as ak
import numpy as np

from picid.data.data_objects.core.metadata_data_object import BaseDataObjectWithMetadata
from picid.data.data_objects.manifest import MetadataManifest
from picid.data.data_objects.slice_info import SliceInfo
from picid.data.data_objects.types import SplitUnitCardinality

T = TypeVar("T", bound=(np.ndarray | ak.Array))
logger = logging.getLogger(__name__)


class DatasetContainer(BaseDataObjectWithMetadata, Generic[T]):
    """
    A container for a complete dataset composed of multiple splits and units.

    It stores data grouped by type (e.g., features, targets), automatically
    normalizes single-unit data into lists, and provides methods for validation
    and transformation.

    Parameters
    ----------
    *args : tuple
        Positional arguments reserved for future compatibility.
    manifest : MetadataManifest | None, optional
        Metadata manifest attached to the container.
    slice_info : SliceInfo | None, optional
        Optional slice metadata describing the current payload window.
    container_metadata : dict[str, Any] | None, optional
        Canonical container-level metadata dictionary.
    unit_metadata : dict[str, Any] | None, optional
        Canonical split-aware unit metadata dictionary.
    metadata : dict[str, Any] | None, optional
        Legacy metadata alias preserved for compatibility.
    **kwargs : Any
        Split-keyed payloads stored under data keys such as ``features``.
    """

    def __init__(
        self,
        *args,
        manifest: Optional[MetadataManifest] = None,
        slice_info: Optional[SliceInfo] = None,
        container_metadata: Optional[dict[str, Any]] = None,
        unit_metadata: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the container and normalize split payloads.

        Parameters
        ----------
        *args : tuple
            Positional arguments reserved for future compatibility.
        manifest : MetadataManifest | None, default=None
            Metadata manifest attached to the container.
        slice_info : SliceInfo | None, default=None
            Optional slice metadata describing the current payload window.
        container_metadata : dict[str, Any] | None, optional
            Canonical container-level metadata dictionary.
        unit_metadata : dict[str, Any] | None, optional
            Canonical split-aware unit metadata dictionary.
        metadata : dict[str, Any] | None, optional
            Legacy metadata alias preserved for compatibility.
        **kwargs : Any
            Split-keyed payloads stored under data keys such as ``features``.
        """
        self.manifest: Optional[MetadataManifest] = (
            manifest if manifest is not None else MetadataManifest()
        )
        self.slice_info: Optional[SliceInfo] = slice_info
        payloads, normalized_container_metadata, normalized_unit_metadata = (
            self._extract_metadata_payloads(
                kwargs=kwargs,
                metadata=metadata,
                container_metadata=container_metadata,
                unit_metadata=unit_metadata,
            )
        )
        self._unit_metadata: dict[str, list[dict[str, Any]]] = normalized_unit_metadata

        super().__init__(
            metadata=normalized_container_metadata,
            **self._normalize_split_payloads(payloads),
        )
        self._assert_internal_list_storage()
        self._assert_unit_metadata_storage()

    @staticmethod
    def _is_split_mapping(value: Any) -> bool:
        """
        Return ``True`` when ``value`` looks like a split-keyed mapping.

        Parameters
        ----------
        value : Any
            Object to inspect.

        Returns
        -------
        bool
            ``True`` when the value has only ``train``, ``val``, or ``test``
            keys.
        """
        if not isinstance(value, Mapping):
            return False
        split_keys = set(value.keys())
        return bool(split_keys) and split_keys.issubset({"train", "val", "test"})

    @classmethod
    def _normalize_container_metadata(
        cls,
        metadata: Optional[dict[str, Any]],
        container_metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Resolve the canonical container-level metadata dictionary.

        Parameters
        ----------
        metadata : dict[str, Any] | None
            Legacy metadata dictionary.
        container_metadata : dict[str, Any] | None
            Canonical metadata dictionary.

        Returns
        -------
        dict[str, Any]
            Normalized container metadata.
        """
        if metadata is not None and container_metadata is not None:
            raise ValueError(
                "Pass either legacy 'metadata' or canonical 'container_metadata', "
                "not both."
            )
        resolved = (
            copy.deepcopy(container_metadata)
            if container_metadata is not None
            else copy.deepcopy(metadata)
            if metadata is not None
            else {}
        )
        if resolved is None:
            resolved = {}
        resolved.setdefault("column_map", {})
        return resolved

    @classmethod
    def _normalize_unit_metadata(
        cls, unit_metadata: Optional[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Normalize explicit or legacy split-aware unit metadata.

        Parameters
        ----------
        unit_metadata : dict[str, Any] | None
            Split-keyed mapping of unit metadata entries.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Canonical split-aware metadata mapping.
        """
        if unit_metadata is None:
            return {}

        if not cls._is_split_mapping(unit_metadata):
            raise TypeError(
                "unit_metadata must be a split-keyed mapping like "
                "{'train': [...], 'val': [...], 'test': [...]}."
            )

        normalized: dict[str, list[dict[str, Any]]] = {}
        for split, split_value in unit_metadata.items():
            if isinstance(split_value, list):
                split_list = split_value
            else:
                split_list = [split_value]
            for item in split_list:
                if not isinstance(item, Mapping):
                    raise TypeError(
                        f"unit_metadata for split '{split}' must contain mapping "
                        f"entries, got {type(item).__name__}."
                    )
            normalized[split] = [dict(item) for item in split_list]
        return normalized

    @classmethod
    def _extract_metadata_payloads(
        cls,
        *,
        kwargs: dict[str, Any],
        metadata: Optional[dict[str, Any]],
        container_metadata: Optional[dict[str, Any]],
        unit_metadata: Optional[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
        """
        Split payload kwargs from canonical container and unit metadata.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Raw constructor keyword arguments.
        metadata : dict[str, Any] | None
            Legacy metadata alias.
        container_metadata : dict[str, Any] | None
            Canonical container metadata.
        unit_metadata : dict[str, Any] | None
            Canonical unit metadata.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]
            Payloads plus normalized container and unit metadata.
        """
        payloads = dict(kwargs)
        legacy_metadata = metadata
        if "metadata" in payloads:
            if legacy_metadata is not None:
                raise ValueError(
                    "metadata was provided both as a named argument and inside kwargs."
                )
            legacy_metadata = payloads.pop("metadata")

        container_metadata_kw = payloads.pop("container_metadata", container_metadata)
        unit_metadata_kw = payloads.pop("unit_metadata", unit_metadata)

        # Legacy loaders overloaded top-level ``metadata`` for both object-level
        # metadata and per-unit split metadata. We disambiguate here once so the
        # rest of the container code can rely on explicit canonical fields.
        if legacy_metadata is not None and cls._is_split_mapping(legacy_metadata):
            if unit_metadata_kw is not None:
                raise ValueError(
                    "Pass either legacy split-aware 'metadata' or canonical "
                    "'unit_metadata', not both."
                )
            normalized_container_metadata = cls._normalize_container_metadata(
                metadata=None,
                container_metadata=container_metadata_kw,
            )
            normalized_unit_metadata = cls._normalize_unit_metadata(legacy_metadata)
        else:
            normalized_container_metadata = cls._normalize_container_metadata(
                metadata=legacy_metadata if isinstance(legacy_metadata, dict) else None,
                container_metadata=container_metadata_kw,
            )
            normalized_unit_metadata = cls._normalize_unit_metadata(unit_metadata_kw)

        return payloads, normalized_container_metadata, normalized_unit_metadata

    @staticmethod
    def _normalize_split_payloads(kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize split payloads so every split stores a list of units.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Raw split-keyed payload mapping passed to the constructor.

        Returns
        -------
        dict[str, Any]
            Normalized payload mapping with list-per-unit storage.
        """
        normalized_kwargs: dict[str, Any] = {}
        for data_key, splits_dict in kwargs.items():
            if isinstance(splits_dict, Mapping):
                normalized_splits = dict(splits_dict)
                for split, data_for_split in normalized_splits.items():
                    if not isinstance(data_for_split, list):
                        logger.info(
                            "Found single item for '%s/%s'. Wrapping it in a list for consistency.",
                            data_key,
                            split,
                        )
                        normalized_splits[split] = [data_for_split]
                normalized_kwargs[data_key] = normalized_splits
            else:
                normalized_kwargs[data_key] = splits_dict
        return normalized_kwargs

    def _iter_split_payloads(
        self, *, include_metadata: bool = True
    ) -> list[tuple[str, Mapping[str, list[Any]]]]:
        """
        Return top-level payloads that expose split-keyed mappings.

        Parameters
        ----------
        include_metadata : bool, default=True
            Whether to include the metadata entry when present.

        Returns
        -------
        list[tuple[str, Mapping[str, list[Any]]]]
            Split-keyed payloads stored in the container.
        """
        payloads: list[tuple[str, Mapping[str, list[Any]]]] = []
        for data_key, splits_data in self.items():
            if data_key == "metadata" and not include_metadata:
                continue
            if isinstance(splits_data, Mapping):
                payloads.append((data_key, splits_data))
        return payloads

    @property
    def unit_metadata(self) -> dict[str, list[dict[str, Any]]]:
        """
        Return per-split metadata aligned with stored units.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Canonical split-aware unit metadata mapping.
        """
        return self._unit_metadata

    def _get_all_split_names(self) -> list[str]:
        """
        Return the union of split names known to payloads and unit metadata.

        Returns
        -------
        list[str]
            Sorted split names present in payloads or unit metadata.
        """
        split_names = {
            split
            for _, splits_data in self._iter_split_payloads(include_metadata=False)
            for split in splits_data.keys()
        }
        split_names.update(self._unit_metadata.keys())
        return sorted(split_names)

    def _unit_metadata_for_split(self, split: str) -> list[dict[str, Any]]:
        """
        Return the stored unit metadata for one split.

        Parameters
        ----------
        split : str
            Split name to query.

        Returns
        -------
        list[dict[str, Any]]
            Metadata entries for the requested split.
        """
        return self._unit_metadata.get(split, [])

    def _assert_internal_list_storage(self) -> None:
        """
        Assert that every split payload is stored as a list of units.
        """
        for data_key, splits_data in self._iter_split_payloads():
            for split, unit_list in splits_data.items():
                assert isinstance(unit_list, list), (
                    f"Expected {data_key}/{split} to be stored as a list of units, "
                    f"got {type(unit_list).__name__}."
                )

    def _assert_unit_metadata_storage(self) -> None:
        """
        Assert canonical split-aware storage for unit metadata.
        """
        for split, unit_list in self._unit_metadata.items():
            assert isinstance(unit_list, list), (
                f"Expected unit_metadata/{split} to be stored as a list, "
                f"got {type(unit_list).__name__}."
            )
            for unit_idx, item in enumerate(unit_list):
                assert isinstance(item, Mapping), (
                    f"Expected unit_metadata/{split}[{unit_idx}] to be a mapping, "
                    f"got {type(item).__name__}."
                )

    def _get_unit_counts_by_split(self) -> dict[str, int]:
        """
        Return one unit count per split after verifying structural alignment.

        Returns
        -------
        dict[str, int]
            Per-split unit counts shared across all payload keys.
        """
        payloads = self._iter_split_payloads(include_metadata=False)
        if not payloads:
            return {}

        reference_key, reference_splits = payloads[0]
        reference_split_names = set(reference_splits.keys())
        split_counts = {
            split: len(unit_list) for split, unit_list in reference_splits.items()
        }

        for data_key, splits_data in payloads[1:]:
            current_split_names = set(splits_data.keys())
            assert current_split_names == reference_split_names, (
                "All payload keys must expose the same split names. "
                f"{reference_key} has {sorted(reference_split_names)} while "
                f"{data_key} has {sorted(current_split_names)}."
            )
            for split, unit_list in splits_data.items():
                current_count = len(unit_list)
                assert split_counts[split] == current_count, (
                    "All payload keys must expose the same number of units per split. "
                    f"{reference_key}/{split} has {split_counts[split]} units while "
                    f"{data_key}/{split} has {current_count}."
                )

        return split_counts

    @property
    def unit_cardinality(self) -> SplitUnitCardinality:
        """
        Describe how many units each populated split contains.

        Returns
        -------
        SplitUnitCardinality
            Cardinality category inferred from populated splits.
        """
        counts = [
            count for count in self._get_unit_counts_by_split().values() if count > 0
        ]
        if not counts:
            return SplitUnitCardinality.EMPTY

        unique_counts = set(counts)
        if unique_counts == {1}:
            return SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT
        if all(count > 1 for count in unique_counts):
            return SplitUnitCardinality.MULTI_UNIT_PER_SPLIT
        return SplitUnitCardinality.MIXED

    def validate(self) -> None:
        """
        Perform comprehensive integrity checks on the dataset structure.
        """
        if not self:
            logger.info("MultiDatasetChunk is empty. Nothing to validate.")
            return
        self._assert_internal_list_storage()
        self._assert_unit_metadata_storage()
        self._get_unit_counts_by_split()

    def copy(self: "DatasetContainer[T]", deep: bool = True) -> "DatasetContainer[T]":
        """
        Return a shallow or deep copy of the container.

        Parameters
        ----------
        deep : bool, default=True
            Whether to deep-copy nested payloads and metadata.

        Returns
        -------
        DatasetContainer[T]
            Copied container.
        """
        new = super().copy(deep=deep)
        manifest_src = getattr(self, "manifest", None)
        if manifest_src is not None:
            object.__setattr__(new, "manifest", manifest_src.copy(deep=deep))
        else:
            object.__setattr__(new, "manifest", None)
        slice_src = getattr(self, "slice_info", None)
        if slice_src is not None:
            object.__setattr__(new, "slice_info", slice_src.copy(deep=deep))
        unit_metadata_src = getattr(self, "_unit_metadata", {})
        object.__setattr__(
            new,
            "_unit_metadata",
            copy.deepcopy(unit_metadata_src) if deep else unit_metadata_src.copy(),
        )
        return new

    def _repr_summary(self) -> str:
        """
        Build a compact summary of keys, splits, and unit counts.

        Returns
        -------
        str
            Safe one-line summary for ``__repr__`` output.
        """
        if not self:
            return "empty"
        parts = []
        for key in list(self.keys())[:5]:
            val = self[key]
            if isinstance(val, Mapping):
                splits = list(val.keys())
                n_units = {s: len(val[s]) for s in splits} if splits else {}
                parts.append(f"{key}=[{','.join(splits)}]({n_units})")
            else:
                parts.append(f"{key}={type(val).__name__}")
        if len(self) > 5:
            parts.append("...")
        return ", ".join(parts)

    def __repr__(self) -> str:
        summary = self._repr_summary()
        cardinality = ""
        if self._iter_split_payloads(include_metadata=False):
            try:
                cardinality = f", unit_cardinality={self.unit_cardinality.value}"
            except (AssertionError, ValueError):
                cardinality = ", unit_cardinality=invalid"
        return f"{self.__class__.__name__}({summary}{cardinality})"
