"""Split-first container views and grouping helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, Generic, List, TypeVar

import awkward as ak
import numpy as np

from picid.data.data_objects.containers.dataset_container import DatasetContainer
from picid.data.data_objects.core.metadata_data_object import BaseDataObjectWithMetadata
from picid.data.data_objects.types import SplitUnitCardinality, SplitViewPolicy
from picid.data.data_objects.validation.split_report import (
    collect_split_alignment_report,
    format_split_alignment_report,
)

T = TypeVar("T", bound=(np.ndarray | ak.Array))
logger = logging.getLogger(__name__)


class SplitDatasetContainer(DatasetContainer, Generic[T]):
    """Specialize :class:`DatasetContainer` with split-first export helpers."""

    def __init__(self, *args, **kwargs: Any):
        """
        Store split payloads in canonical list-per-unit form.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to :class:`DatasetContainer`.
        **kwargs : dict[str, Any]
            Split-keyed payloads stored under data keys such as ``features`` or
            ``target``.
        """
        super().__init__(*args, **kwargs)

    def _collect_split_alignment_report(self) -> dict[str, Any]:
        """Collect a non-strict report about split-key alignment and unit scales.

        Returns
        -------
        dict[str, Any]
            Diagnostic report for the current split payload structure.
        """
        return collect_split_alignment_report(
            self._iter_split_payloads(include_metadata=False)
        )

    def _collect_unit_metadata_alignment(self) -> dict[str, Any]:
        """Collect split/count alignment diagnostics for canonical unit metadata."""
        if not self.unit_metadata:
            return {
                "unit_metadata_match": True,
                "unit_metadata_mismatches": {},
            }

        payload_splits = self._get_all_split_names()
        try:
            expected_counts = self._get_aligned_unit_counts_by_split_payload_only()
        except AssertionError:
            expected_counts = {}

        mismatches: dict[str, dict[str, int]] = {}
        for split in payload_splits:
            actual = len(self._unit_metadata_for_split(split))
            expected = expected_counts.get(split)
            if expected is None:
                # Relaxed validation still reports unit_metadata problems even
                # when payload keys disagree on splits. In that case we only
                # compare against a split-local payload size when it is
                # unambiguous, instead of inventing a reference count.
                lengths = {
                    len(splits_data[split])
                    for _, splits_data in self._iter_split_payloads(
                        include_metadata=False
                    )
                    if split in splits_data
                }
                if len(lengths) == 1:
                    expected = lengths.pop()
            if expected is None:
                continue
            if actual != expected:
                mismatches[split] = {"expected": expected, "actual": actual}

        return {
            "unit_metadata_match": not mismatches,
            "unit_metadata_mismatches": mismatches,
        }

    def _format_split_alignment_report(self, report: dict[str, Any]) -> str:
        """Format a compact ASCII table for split alignment diagnostics.

        Parameters
        ----------
        report : dict[str, Any]
            Structured alignment report returned by
            :meth:`_collect_split_alignment_report`.

        Returns
        -------
        str
            Human-readable table representation of the report.
        """
        return format_split_alignment_report(report)

    @property
    def unit_cardinality(self) -> SplitUnitCardinality:
        """Describe per-split unit cardinality without requiring cross-key alignment.

        Returns
        -------
        SplitUnitCardinality
            Cardinality category inferred from populated split payloads.
        """
        counts = [
            len(unit_list)
            for _, splits_data in self._iter_split_payloads(include_metadata=False)
            for unit_list in splits_data.values()
            if len(unit_list) > 0
        ]
        if not counts:
            return SplitUnitCardinality.EMPTY

        unique_counts = set(counts)
        if unique_counts == {1}:
            return SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT
        if all(count > 1 for count in unique_counts):
            return SplitUnitCardinality.MULTI_UNIT_PER_SPLIT
        return SplitUnitCardinality.MIXED

    def validate(self, strict: bool = False) -> dict[str, Any]:
        """Validate split payloads and report post-transform heterogeneity.

        Parameters
        ----------
        strict : bool, default=False
            Whether to raise on heterogeneous payload structure.

        Returns
        -------
        dict[str, Any]
            Structured validation report.
        """
        self._assert_internal_list_storage()

        report = self._collect_split_alignment_report()
        report.update(self._collect_unit_metadata_alignment())
        report["is_consistent"] = (
            report["is_consistent"] and report["unit_metadata_match"]
        )
        if not report["rows"]:
            logger.info("SplitDatasetContainer is empty. Nothing to validate.")
            return report

        table = self._format_split_alignment_report(report)
        if not report["unit_metadata_match"]:
            mismatch_lines = [
                "unit_metadata_mismatches:",
                *(
                    f"  {split}: expected {values['expected']} entries but got "
                    f"{values['actual']}"
                    for split, values in report["unit_metadata_mismatches"].items()
                ),
            ]
            table = f"{table}\n\n" + "\n".join(mismatch_lines)
        if report["is_consistent"]:
            logger.info("SplitDatasetContainer validation successful.\n%s", table)
            return report

        message = (
            "SplitDatasetContainer contains heterogeneous split payloads.\n" f"{table}"
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)
        return report

    def _get_aligned_unit_counts_by_split(self) -> dict[str, int]:
        """Return strict split counts for operations that require aligned keys.

        Returns
        -------
        dict[str, int]
            Per-split unit counts after strict cross-key alignment checks.
        """
        split_counts = self._get_aligned_unit_counts_by_split_payload_only()
        self._assert_unit_metadata_alignment(split_counts)
        return split_counts

    def _get_aligned_unit_counts_by_split_payload_only(self) -> dict[str, int]:
        """Return strict split counts across payload keys, ignoring unit metadata."""
        return DatasetContainer._get_unit_counts_by_split(self)

    def _assert_unit_metadata_alignment(
        self, split_counts: dict[str, int] | None = None
    ) -> None:
        """Raise when unit metadata is not aligned with split unit counts."""
        if not self.unit_metadata:
            return
        if split_counts is None:
            split_counts = self._get_aligned_unit_counts_by_split_payload_only()
        mismatches = {}
        for split in self._get_all_split_names():
            # unit_metadata is split-scoped and must stay 1:1 with the unit list
            # for every populated split. This is the contract the transform
            # pipeline relies on when it attaches metadata to chunk inputs.
            expected = split_counts.get(split, 0)
            actual = len(self._unit_metadata_for_split(split))
            if actual != expected:
                mismatches[split] = {"expected": expected, "actual": actual}
        if mismatches:
            mismatch_str = ", ".join(
                f"{split}: expected {values['expected']} got {values['actual']}"
                for split, values in mismatches.items()
            )
            raise ValueError(
                f"unit_metadata is misaligned with payload units ({mismatch_str})."
            )

    def validate_strict(self) -> dict[str, Any]:
        """Validate split payloads with strict cross-key alignment enabled.

        Returns
        -------
        dict[str, Any]
            Structured validation report.
        """
        report = self.validate(strict=False)
        self._get_aligned_unit_counts_by_split()
        return report

    def to_split_dict(
        self,
        view_policy: SplitViewPolicy = SplitViewPolicy.KEEP_UNIT_LISTS,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reorganize the data into a dictionary grouped by split.

        Parameters
        ----------
        view_policy : SplitViewPolicy, default=SplitViewPolicy.KEEP_UNIT_LISTS
            Export mode controlling whether singleton unit lists are preserved.

        Returns
        -------
        dict[str, dict[str, Any]]
            Split-first view of the stored payloads.
        """
        if not self:
            return {}

        all_splits = self._get_all_split_names()

        if view_policy == SplitViewPolicy.UNWRAP_SINGLETONS:
            if self.unit_cardinality != SplitUnitCardinality.SINGLE_UNIT_PER_SPLIT:
                raise ValueError(
                    "SplitViewPolicy.UNWRAP_SINGLETONS requires exactly one unit in "
                    "every populated split."
                )

        split_dict = {split: {} for split in all_splits}
        for data_key, splits_data in self._iter_split_payloads():
            for split, unit_list in splits_data.items():
                if view_policy == SplitViewPolicy.UNWRAP_SINGLETONS and unit_list:
                    split_dict[split][data_key] = unit_list[0]
                else:
                    split_dict[split][data_key] = unit_list
        for split in all_splits:
            for data_key, splits_data in self._iter_split_payloads():
                if data_key not in split_dict[split]:
                    split_dict[split][data_key] = []
            split_dict[split]["unit_metadata"] = self._unit_metadata_for_split(split)

        return split_dict

    def group_by_split(self) -> Dict[str, List[BaseDataObjectWithMetadata]]:
        """
        Reorganize the data into a dictionary of data-object lists, grouped by split.

        Returns
        -------
        dict[str, list[BaseDataObjectWithMetadata]]
            Per-split lists of unit-level data objects.
        """
        if not self:
            return {}

        self._get_aligned_unit_counts_by_split()

        splits, num_units_per_split, data_keys = set(), {}, list(self.keys())
        for key in data_keys:
            if isinstance(self[key], Mapping):
                splits.update(self[key].keys())
        splits.update(self.unit_metadata.keys())

        for split in splits:
            lengths = {
                len(self[key][split])
                for key in data_keys
                if isinstance(self[key], Mapping) and split in self[key]
            }
            if lengths:
                num_units_per_split[split] = lengths.pop()

        grouped_dict: Dict[str, List[BaseDataObjectWithMetadata]] = {
            split: [] for split in splits
        }
        for split in splits:
            num_units = num_units_per_split.get(split, 0)
            for unit_idx in range(num_units):
                unit_data = {
                    data_key: self[data_key][split][unit_idx]
                    for data_key in data_keys
                    if isinstance(self[data_key], Mapping) and split in self[data_key]
                }
                if unit_data:
                    unit_metadata = self._unit_metadata_for_split(split)
                    grouped_dict[split].append(
                        BaseDataObjectWithMetadata(
                            metadata=(
                                unit_metadata[unit_idx]
                                if unit_idx < len(unit_metadata)
                                else None
                            ),
                            **unit_data,
                        )
                    )
        return grouped_dict
