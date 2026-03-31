"""
Concrete pipeline step implementations.

The transform pipeline runs these steps in order (see strategy.py for the
sequence). Each step receives TransformContext, reads or mutates it, and
returns it. Data flow in short:

  1. CopyStep: shallow-copy data, set transformed_data and available_splits.
  2. RegisterChunkBuilderStep: set context.build_fit_chunks / build_transform_chunks.
  3. IntegrityCheckBeforeStep: run before-checks on chunks per split.
  4. FitStep: call transform.fit_multi_source on fit chunks.
  5. TransformStep: call transform.transform_multi_source per split.
  6. PostprocessStep: run postprocess_fn on raw results, fill transformed_results_for_new_key.
  7. MergeStep: strategy._merge_transformed_data into context.transformed_data.
  8. IntegrityCheckAfterStep: run after-checks on assign_to slice.
  9. RecordManifestStep: append a manifest entry if transformed_data has a manifest.

Chunk builders are provided by the strategy and produce List[NamedTransformInput]
per split; they hold references to underlying arrays (no copy of array data).
"""

from __future__ import annotations

import copy
import datetime
import logging
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import awkward as ak

from picid.data.data_objects import NamedTransformInput
from picid.data.data_objects import SplitDatasetContainer
from picid.data.data_objects.slice_info import SliceInfo
from picid.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from picid.transforms.base.integrity import run_checks

from picid.transforms.base.pipeline.checks import _run_scoped_checks
from picid.transforms.base.pipeline.context import TransformContext

logger = logging.getLogger(__name__)


def _slice_info_to_metadata(
    slice_info: Optional[SliceInfo],
) -> Optional[Dict[str, Any]]:
    """
    Serialize :class:`SliceInfo` to a dict for inclusion in transform metadata.

    Parameters
    ----------
    slice_info : SliceInfo or None
        Slice descriptor to serialize.

    Returns
    -------
    dict[str, Any] or None
        Serialized metadata, or ``None`` when ``slice_info`` is absent.
    """
    if slice_info is None:
        return None
    return {
        "split": slice_info.split,
        "unit_ids": slice_info.unit_ids,
        "cycle_ids": slice_info.cycle_ids,
        "bounds": slice_info.bounds,
        "index_map": slice_info.index_map,
    }


class CopyStep:
    """
    First step: copy container and keys, determine available splits.

    Sets context.transformed_data (shallow copy of context.data for relevant keys),
    context.available_splits (from apply_to key, optionally filtered by
    transform_on_keys), and initialises transformed_results_for_new_key and
    _raw_transformed_by_split. Awkward arrays are copied with deep=False to avoid
    full deep copies.
    """

    def run(self, context: TransformContext) -> TransformContext:
        data = context.data
        apply_to_keys = context.apply_to_keys
        assign_to_keys = context.assign_to_keys or []

        if not apply_to_keys:
            raise ValueError("apply_to_keys must not be empty.")

        # Use first apply_to key to discover splits; data[apply_key] must be split-keyed.
        apply_key = apply_to_keys[0]
        if apply_key not in data:
            raise KeyError(
                f"apply_to key '{apply_key}' not found in data. "
                f"Available: {sorted(str(k) for k in data.keys())}"
            )
        if not isinstance(data[apply_key], Mapping):
            raise TypeError(f"data['{apply_key}'] must be a split-keyed mapping.")

        # Shallow copy; then per-key copy: ak.Array uses deep=False, others deep=True.
        transformed_data = data.copy(deep=False)
        for key in set(list(apply_to_keys) + list(assign_to_keys)):
            if key in data:
                first_leaf = self._first_leaf(data[key])
                if isinstance(first_leaf, ak.Array):
                    transformed_data[key] = data[key].copy(deep=False)
                else:
                    transformed_data[key] = data[key].copy(deep=True)

        available_splits = list(data[apply_key].keys())
        # Optionally restrict to only certain splits (e.g. train/val, not test).
        if context.transform_on_keys is not None:
            available_splits = list(
                set(available_splits) & set(context.transform_on_keys)
            )
            if not available_splits:
                transform_name = type(context.transform_instance).__name__
                raise ValueError(
                    f"Transform {transform_name!r}: no splits remain after "
                    f"applying transform_on_keys={context.transform_on_keys}. "
                    f"Data splits: {list(data[apply_key].keys())}."
                )

        context.transformed_data = transformed_data
        context.available_splits = available_splits
        context.transformed_results_for_new_key = defaultdict(dict)
        context._raw_transformed_by_split = {}
        return context

    @staticmethod
    def _first_leaf(val: Any) -> Any:
        """
        Drill down until the first non-container value is found.

        Parameters
        ----------
        val : Any
            Potentially nested split mapping or list.

        Returns
        -------
        Any
            First scalar-like leaf value used to detect array container type.
        """
        while isinstance(val, (Mapping, list)):
            val = next(iter(val.values())) if isinstance(val, Mapping) else val[0]
        return val


class RegisterChunkBuilderStep:
    """
    Register callables that build chunks per split for fit and transform.

    Sets context.build_fit_chunks() and context.build_transform_chunks(split).
    These are used by IntegrityCheckBeforeStep, FitStep, and TransformStep.
    If no strategy is set, leaves context unchanged (steps will build inline when needed).
    """

    def run(self, context: TransformContext) -> TransformContext:
        if context.strategy is None:
            return context

        strategy = context.strategy
        data = context.data
        apply_to_keys = context.apply_to_keys
        fit_on_split = context.fit_on_split
        fit_on_key = context.fit_on_key

        def build_transform_chunks(split: str) -> List[NamedTransformInput]:
            target_data = {key: data[key] for key in apply_to_keys}
            return strategy._prepare_chunks_for_split(
                target_data,
                split,
                # unit_metadata is optional; when present it is threaded into
                # the split-local chunk builder so every NamedTransformInput
                # sees its own per-unit metadata.
                unit_metadata_by_split=(
                    data.unit_metadata if getattr(data, "unit_metadata", {}) else None
                ),
            )

        def build_fit_chunks() -> List[NamedTransformInput]:
            if not fit_on_split or not fit_on_key:
                return []
            fit_data = {fit_on_key: data[fit_on_key]}
            return strategy._prepare_chunks_for_split(
                fit_data,
                fit_on_split,
                unit_metadata_by_split=(
                    data.unit_metadata if getattr(data, "unit_metadata", {}) else None
                ),
            )

        context.build_transform_chunks = build_transform_chunks
        context.build_fit_chunks = build_fit_chunks
        return context


class IntegrityCheckStep:
    """
    Legacy step: run optional no-arg integrity checks from context.integrity_checks.

    Deprecated in favour of IntegrityCheckBeforeStep / IntegrityCheckAfterStep
    with scoped (name, fn, policy) checks. Still supported for backward compatibility.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if not context.integrity_checks:
            return context
        run_checks(context.integrity_checks)
        return context


class IntegrityCheckBeforeStep:
    """
    Run before-checks on each split's transform chunks.

    Uses context.integrity_checks_before (e.g. no NaNs, length consistency).
    Runs _run_scoped_checks from checks.py per split; records ran check names
    and sets integrity_before_ran for RecordManifestStep.
    """

    def run(self, context: TransformContext) -> TransformContext:
        checks = context.integrity_checks_before
        if not checks:
            return context

        build = context.build_transform_chunks
        if build is None or context.available_splits is None:
            return context

        ran: List[str] = []
        step_id = context.step_id or type(context.transform_instance).__name__
        for split in context.available_splits:
            split_chunks = {split: build(split)}
            ran_for_split = _run_scoped_checks(
                checks,
                split_chunks,
                step_id=step_id,
                split=split,
                stage="before-transform",
            )
            for name in ran_for_split:
                if name not in ran:
                    ran.append(name)

        context.integrity_before_check_names = ran
        context.integrity_before_ran = True
        return context


class FitStep:
    """
    Run the transform's fit on fit chunks (one split, one key).

    Uses context.build_fit_chunks() if set, otherwise builds chunks via strategy
    for fit_on_split/fit_on_key. Always calls fit_multi_source(list of chunks).
    """

    def run(self, context: TransformContext) -> TransformContext:
        requires_fit = getattr(context.transform_instance, "requires_fit", False)

        if not context.fit_on_split or not context.fit_on_key:
            if requires_fit:
                transform_name = type(context.transform_instance).__name__
                raise ValueError(
                    f"Transform {transform_name!r} requires fitting but "
                    "fit_on_split and/or fit_on_key are not set."
                )
            return context

        build = context.build_fit_chunks
        if build is not None:
            fit_chunks = build()
        elif context.strategy is not None:
            logger.debug("FitStep: build_fit_chunks not registered. Building inline.")
            fit_data = {context.fit_on_key: context.data[context.fit_on_key]}
            fit_chunks = context.strategy._prepare_chunks_for_split(
                fit_data,
                context.fit_on_split,
                unit_metadata_by_split=(
                    context.data.unit_metadata
                    if getattr(context.data, "unit_metadata", {})
                    else None
                ),
            )
        else:
            return context

        if not fit_chunks:
            return context

        metadata = {
            "mode": context.fit_on_split,
            "apply_to_keys": context.fit_on_key,
            "assign_to_keys": context.assign_to_keys,
            "assign_to_map": context.assign_to_keys_map,
            "container_metadata": getattr(context.data, "container_metadata", {}),
        }
        if context.include_slice_info_in_metadata and context.slice_info is not None:
            metadata["slice_info"] = _slice_info_to_metadata(context.slice_info)

        context.transform_instance.fit_multi_source(fit_chunks, metadata=metadata)
        return context


class TransformStep:
    """
    Run the transform on each split's chunks and store raw results.

    For each split: get chunks via build_transform_chunks(split) or strategy;
    call transform_multi_source(chunks); store (transformed_chunks, metadata)
    in context._raw_transformed_by_split. PostprocessStep and MergeStep consume
    this later.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if context.available_splits is None:
            return context

        build = context.build_transform_chunks

        for split in context.available_splits:
            if build is not None:
                chunks = build(split)
            elif context.strategy is not None:
                logger.debug(
                    "TransformStep: build_transform_chunks not registered. "
                    "Building inline for split '%s'.",
                    split,
                )
                target_data = {key: context.data[key] for key in context.apply_to_keys}
                chunks = context.strategy._prepare_chunks_for_split(
                    target_data,
                    split,
                    unit_metadata_by_split=(
                        context.data.unit_metadata
                        if getattr(context.data, "unit_metadata", {})
                        else None
                    ),
                )
            else:
                continue

            if not chunks:
                context.log[split] = {}
                continue

            metadata = {
                "mode": split,
                "apply_to_keys": context.apply_to_keys,
                "assign_to_keys": context.assign_to_keys,
                "assign_to_map": context.assign_to_keys_map,
                "container_metadata": getattr(context.data, "container_metadata", {}),
            }
            if (
                context.include_slice_info_in_metadata
                and context.slice_info is not None
            ):
                metadata["slice_info"] = _slice_info_to_metadata(context.slice_info)

            transformed_chunks, log = context.transform_instance.transform_multi_source(
                chunks, metadata=metadata
            )
            context.log[split] = log
            context._raw_transformed_by_split[split] = (transformed_chunks, metadata)

        return context


class PostprocessStep:
    """
    Convert raw transform output into assign_to-keyed structure.

    For each split: run context.postprocess_fn(transformed_chunks, metadata)
    (e.g. postprocess_transformed_data from strategy), then map each
    assign_to_map key from the postprocessed chunks into
    context.transformed_results_for_new_key[assign_to_key][split].
    MergeStep then merges these into context.transformed_data.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if (
            context.postprocess_fn is None
            or context.transformed_results_for_new_key is None
        ):
            return context

        for split, (
            transformed_chunks,
            metadata,
        ) in context._raw_transformed_by_split.items():
            postprocessed = context.postprocess_fn(transformed_chunks, metadata)
            for assign_to_key, assign_to_map in zip(
                context.assign_to_keys, context.assign_to_keys_map
            ):
                for chunk in postprocessed:
                    if assign_to_map not in chunk:
                        raise KeyError(
                            f"Transformed chunk missing expected key '{assign_to_map}'. "
                            f"Available: {list(chunk.keys())}"
                        )
                context.transformed_results_for_new_key[assign_to_key][split] = [
                    chunk[assign_to_map] for chunk in postprocessed
                ]
        return context


class MergeStep:
    """
    Merge postprocessed results back into context.transformed_data.

    Delegates to strategy._merge_transformed_data with transformed_results_for_new_key
    and available_splits. Logs warnings if type or unit count changed for an
    assign_to key that already existed in the original data.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if (
            context.strategy is None
            or context.transformed_data is None
            or context.available_splits is None
            or context.transformed_results_for_new_key is None
        ):
            return context

        self._propagate_unit_metadata(context)
        self._warn_on_changes(context)
        self._assert_unit_metadata_alignment(context)

        context.strategy._merge_transformed_data(
            base_data=context.transformed_data,
            transformed_results=context.transformed_results_for_new_key,
            available_splits=context.available_splits,
        )
        return context

    @staticmethod
    def _propagate_unit_metadata(context: TransformContext) -> None:
        """
        Let the transform decide how unit metadata should follow the output.

        Parameters
        ----------
        context : TransformContext
            Pipeline context that holds the transformed payload and raw
            transformed results.
        """
        transformed_data = context.transformed_data
        if transformed_data is None:
            return

        unit_metadata = getattr(transformed_data, "unit_metadata", {})
        if not unit_metadata:
            return

        transform = context.transform_instance
        propagate_hook = getattr(transform, "propagate_unit_metadata", None)
        if propagate_hook is None:
            return

        # The hook runs before the alignment assertion on purpose. At this point
        # we already know the transformed payload shape, but we have not yet
        # rejected mismatched metadata. This is the one place where a
        # cardinality-changing transform can replace stale per-unit metadata with
        # metadata that truthfully describes the transformed output.
        hook_metadata = {
            "transform_name": context.step_id or type(transform).__name__,
            "apply_to_keys": context.apply_to_keys,
            "assign_to_keys": context.assign_to_keys,
            "assign_to_map": context.assign_to_keys_map,
            "available_splits": context.available_splits,
            "container_metadata": getattr(context.data, "container_metadata", {}),
        }

        propagated = propagate_hook(
            unit_metadata_by_split=unit_metadata,
            transformed_results_for_new_key=context.transformed_results_for_new_key,
            metadata=hook_metadata,
        )

        if propagated is None:
            return
        if not isinstance(propagated, Mapping):
            raise TypeError(
                f"Transform '{type(transform).__name__}' returned invalid "
                f"unit_metadata propagation output of type {type(propagated).__name__}; "
                "expected a split-keyed mapping or None."
            )

        if propagated:
            normalized = SplitDatasetContainer._normalize_unit_metadata(
                dict(propagated)
            )
        else:
            normalized = {}

        # The transformed container is a shallow copy of the input container,
        # so replacing the private storage here is safe and keeps the source
        # container untouched. We intentionally mutate only the copied
        # ``transformed_data`` view that continues through the pipeline.
        transformed_data._unit_metadata = copy.deepcopy(normalized)

    @staticmethod
    def _warn_on_changes(context: TransformContext) -> None:
        """
        Log warnings when an assign_to key's type or length changes.

        Parameters
        ----------
        context : TransformContext
            Pipeline context with both original and transformed payloads.
        """
        transform_name = type(context.transform_instance).__name__
        for assign_to_key in context.transformed_results_for_new_key:
            if assign_to_key not in context.data:
                continue
            for split in context.available_splits:
                before = context.data[assign_to_key].get(split)
                after_list = context.transformed_results_for_new_key[assign_to_key].get(
                    split
                )
                if before is None or after_list is None:
                    continue
                if not isinstance(after_list, type(before)):
                    logger.warning(
                        "Transform '%s': type of '%s/%s' changed from %s to %s.",
                        transform_name,
                        assign_to_key,
                        split,
                        type(before).__name__,
                        type(after_list).__name__,
                    )
                if isinstance(before, list) and isinstance(after_list, list):
                    if len(before) != len(after_list):
                        logger.warning(
                            "Transform '%s': unit count of '%s/%s' changed "
                            "from %d to %d.",
                            transform_name,
                            assign_to_key,
                            split,
                            len(before),
                            len(after_list),
                        )

    @staticmethod
    def _assert_unit_metadata_alignment(context: TransformContext) -> None:
        """
        Raise when transformed output can no longer align with unit metadata.

        Parameters
        ----------
        context : TransformContext
            Pipeline context holding transformed results and unit metadata.
        """
        transformed_data = context.transformed_data
        if transformed_data is None:
            return

        unit_metadata = getattr(transformed_data, "unit_metadata", {})
        if not unit_metadata:
            return

        mismatches = []
        for (
            assign_to_key,
            split_results,
        ) in context.transformed_results_for_new_key.items():
            for split, after_list in split_results.items():
                # Unit metadata is preserved automatically only for
                # unit-preserving transforms. If a transform collapses or expands
                # units, it must implement an explicit propagation strategy
                # instead of silently leaving metadata behind.
                expected = len(unit_metadata.get(split, []))
                actual = len(after_list)
                if expected != actual:
                    mismatches.append(
                        f"{assign_to_key}/{split}: expected {expected} metadata "
                        f"entries but got {actual} transformed units"
                    )

        if mismatches:
            raise ValueError(
                "Transform output cannot preserve unit_metadata alignment: "
                + "; ".join(mismatches)
                + ". If this transform changes unit cardinality, implement "
                "`propagate_unit_metadata(...)` on the transform and return one of "
                "`preserve_unit_metadata(...)`, `aggregate_unit_metadata(...)`, or "
                "`drop_unit_metadata(...)` before this merge step. Preserve is the "
                "default for unit-preserving transforms; aggregate and drop are "
                "the escape hatches for transforms that intentionally reshape "
                "unit identity."
            )


class IntegrityCheckAfterStep:
    """
    Run after-checks on the assign_to slice of transformed data.

    Builds assign_to_slice from context.transformed_data for assign_to_keys,
    runs context.integrity_checks_after (e.g. structure, split preservation),
    and records ran check names for RecordManifestStep.
    """

    def run(self, context: TransformContext) -> TransformContext:
        checks = context.integrity_checks_after
        if not checks or context.transformed_data is None:
            return context

        for key in context.assign_to_keys:
            if key not in context.transformed_data:
                raise KeyError(
                    f"Transformed data from strategy does not contain '{key}' key at the top level."
                )

        assign_to_slice: Dict[str, Any] = {
            key: context.transformed_data[key]
            for key in context.assign_to_keys
            if key in context.transformed_data
        }

        step_id = context.step_id or type(context.transform_instance).__name__
        ran = _run_scoped_checks(
            checks, assign_to_slice, step_id=step_id, stage="after-transform"
        )
        context.integrity_after_check_names = ran
        context.integrity_after_ran = True
        return context


def _get_producer_version() -> str:
    """
    Return the picid package version for manifest entries.

    Returns
    -------
    str
        Installed picid package version, or ``"0.0.0"`` if metadata lookup fails.
    """
    try:
        import importlib.metadata

        return importlib.metadata.version("picid")
    except Exception:
        return "0.0.0"


class RecordManifestStep:
    """
    Append a transform manifest entry if the container has a MetadataManifest.

    Writes transform name, apply_to/assign_to keys, fit info, integrity run
    status, timing, and optional slice_info so runs are auditable.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if context.transformed_data is None:
            return context
        manifest = getattr(context.transformed_data, "manifest", None)
        if manifest is None or not isinstance(manifest, MetadataManifest):
            return context

        step_id = context.step_id or type(context.transform_instance).__name__

        payload: Dict[str, Any] = {
            "transform_name": step_id,
            "transform_class": (
                type(context.transform_instance).__name__
                if context.transform_instance
                else None
            ),
            "apply_to_keys": context.apply_to_keys,
            "assign_to_keys": context.assign_to_keys,
            "fit_on_split": context.fit_on_split,
            "fit_on_key": context.fit_on_key,
            "log_splits": list(context.log.keys()) if context.log else [],
            "integrity_before_run": context.integrity_before_ran,
            "integrity_before_checks": context.integrity_before_check_names,
            "integrity_after_run": context.integrity_after_ran,
            "integrity_after_checks": context.integrity_after_check_names,
        }

        if context.started_at is not None:
            payload["started_at"] = context.started_at
            payload["started_at_iso"] = datetime.datetime.fromtimestamp(
                context.started_at, tz=datetime.timezone.utc
            ).isoformat()
        if context.finished_at is not None:
            payload["finished_at"] = context.finished_at
            payload["finished_at_iso"] = datetime.datetime.fromtimestamp(
                context.finished_at, tz=datetime.timezone.utc
            ).isoformat()
        if context.started_at is not None and context.finished_at is not None:
            payload["duration_seconds"] = round(
                context.finished_at - context.started_at, 6
            )
        if context.slice_info is not None:
            payload["slice_info"] = {
                "split": context.slice_info.split,
                "n_units": (
                    len(context.slice_info.unit_ids)
                    if context.slice_info.unit_ids
                    else None
                ),
                "n_cycles": (
                    len(context.slice_info.cycle_ids)
                    if context.slice_info.cycle_ids
                    else None
                ),
            }

        split_str = (
            ",".join(context.available_splits) if context.available_splits else None
        )
        entry = ManifestEntry(
            schema_version=MANIFEST_SCHEMA_VERSION,
            producer_version=_get_producer_version(),
            category="transform",
            payload=payload,
            step_id=step_id,
            key=context.apply_to_keys[0] if context.apply_to_keys else None,
            split=split_str,
        )
        manifest.add(entry)
        return context
