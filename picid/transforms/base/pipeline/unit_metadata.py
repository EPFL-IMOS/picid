"""
Helpers for transform-level unit_metadata propagation.

The pipeline treats ``unit_metadata`` as split-scoped, so transforms that keep
the same unit cardinality can leave it untouched, while transforms that change
cardinality must explicitly decide how to propagate it.

The helpers in this module implement the three common strategies:

* preserve: keep metadata unchanged
* aggregate: collapse many source units into one summary record
* drop: intentionally discard unit metadata

The pipeline calls transform-owned hooks before the alignment assertion. Hooks
may return a replacement ``unit_metadata`` mapping or ``None`` to keep the
current metadata unchanged.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

UnitMetadataBySplit = dict[str, list[dict[str, Any]]]


def preserve_unit_metadata(
    *, unit_metadata_by_split: Mapping[str, list[dict[str, Any]]]
) -> UnitMetadataBySplit:
    """
    Return a deep-copied metadata mapping without changing its semantics.

    Parameters
    ----------
    unit_metadata_by_split : Mapping[str, list[dict[str, Any]]]
        Current metadata grouped by split.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        A deep copy of the provided mapping.
    """
    # Deep-copy here so transforms can safely reuse this helper without sharing
    # mutable metadata objects with the input container.
    return copy.deepcopy(dict(unit_metadata_by_split))


def drop_unit_metadata(
    *, unit_metadata_by_split: Mapping[str, list[dict[str, Any]]]
) -> UnitMetadataBySplit:
    """
    Intentionally remove all unit metadata from the transformed payload.

    Parameters
    ----------
    unit_metadata_by_split : Mapping[str, list[dict[str, Any]]]
        Current metadata grouped by split.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        An empty mapping, signalling that unit metadata should be discarded.
    """
    # Returning an empty mapping makes the pipeline treat unit_metadata as absent.
    # This should be used only when unit identity is genuinely no longer
    # meaningful after the transform, not merely as a shortcut around alignment.
    return {}


def aggregate_unit_metadata(
    *,
    unit_metadata_by_split: Mapping[str, list[dict[str, Any]]],
    transformed_results_for_new_key: Mapping[str, Mapping[str, list[Any]]],
    metadata: Mapping[str, Any] | None = None,
) -> UnitMetadataBySplit:
    """
    Collapse per-unit metadata into one summary entry per transformed split.

    This is the expected strategy for transforms that turn many units into one
    split-level dense output. The helper keeps untouched splits unchanged and
    only rewrites the splits that appear in ``transformed_results_for_new_key``.

    Parameters
    ----------
    unit_metadata_by_split : Mapping[str, list[dict[str, Any]]]
        Current metadata grouped by split before propagation.
    transformed_results_for_new_key : Mapping[str, Mapping[str, list[Any]]]
        Postprocessed transformed values grouped by assign-to key and split.
    metadata : Mapping[str, Any], optional
        Pipeline metadata that can be threaded into the aggregated entry for
        provenance.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        A split-keyed mapping with one aggregated metadata entry per transformed split.
    """
    propagated: UnitMetadataBySplit = copy.deepcopy(dict(unit_metadata_by_split))
    metadata = dict(metadata or {})
    transform_name = metadata.get("transform_name")

    # A transform may write multiple assign_to keys in one step (for example
    # ``features`` and ``anomaly_detection``). Aggregation is only safe if every
    # transformed key agrees on how many output items exist for a split.
    split_lengths: dict[str, set[int]] = defaultdict(set)
    for split_results in transformed_results_for_new_key.values():
        for split, after_list in split_results.items():
            split_lengths[split].add(len(after_list))

    for split, lengths in split_lengths.items():
        if len(lengths) != 1:
            raise ValueError(
                f"Cannot aggregate unit_metadata for split '{split}': "
                f"inconsistent transformed lengths {sorted(lengths)}."
            )

        transformed_count = next(iter(lengths))
        source_metadata = list(unit_metadata_by_split.get(split, []))

        # Empty outputs are rare, but we keep the split present so downstream
        # alignment checks can still reason about the transformed payload.
        if transformed_count == 0:
            propagated[split] = []
            continue

        if transformed_count != 1:
            raise ValueError(
                f"Cannot aggregate unit_metadata for split '{split}': "
                f"expected a single transformed unit but got {transformed_count}."
            )

        # We intentionally produce truthful metadata for the new post-transform
        # object rather than pretending the original per-unit entries still
        # align. Downstream consumers can still inspect provenance through the
        # preserved source names and ids.
        aggregated_entry: dict[str, Any] = {
            "unit_name": f"aggregated::{split}",
            "unit_id": None,
            "metadata_hook": "aggregate",
            "source_unit_count": len(source_metadata),
            "source_unit_names": [
                entry.get("unit_name")
                for entry in source_metadata
                if isinstance(entry, Mapping)
            ],
            "source_unit_ids": [
                entry.get("unit_id")
                for entry in source_metadata
                if isinstance(entry, Mapping)
            ],
        }
        if transform_name is not None:
            aggregated_entry["transform_name"] = transform_name

        propagated[split] = [aggregated_entry]

    return propagated
