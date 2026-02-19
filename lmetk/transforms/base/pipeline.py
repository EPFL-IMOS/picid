"""
pipeline.py
===========

Composable pipeline steps and lifecycle hooks for the transform pipeline.

Pipeline order (see TransformStrategy.apply):
    CopyStep
    → RegisterChunkBuilderStep    (stores callables on context; no data materialised)
    → IntegrityCheckBeforeStep    (calls builder per-split; scoped to apply_to chunks)
    → FitStep                     (calls build_fit_chunks(); one call, GC'd on return)
    → TransformStep               (calls build_transform_chunks(split) per split; streamed)
    → PostprocessStep
    → MergeStep
    → IntegrityCheckAfterStep     (scoped to assign_to slice of transformed_data)
    → RecordManifestStep

Memory design
-------------
Chunks are never stored on context. RegisterChunkBuilderStep stores two callables:
    context.build_fit_chunks()        → List[NamedTransformInput]  (fit split only)
    context.build_transform_chunks(split) → List[NamedTransformInput]  (one split)

Each consumer calls the builder at the moment it needs data and holds a local
reference only. When the local goes out of scope the chunks are eligible for GC
before the next split is built. For PHM datasets (large time-series, many units)
this keeps peak memory to one split's worth of chunk wrappers, not all splits.

Note: chunk wrappers hold *references* to the underlying arrays in context.data —
no array data is copied by the builders.

Integrity check function signatures
------------------------------------
Before-checks receive Dict[str, List[NamedTransformInput]] for a single split.
After-checks receive Dict[str, Any] — the assign_to slice of transformed_data.
This enforces scope at the API boundary rather than by convention.

Legacy no-arg check functions are supported via introspection with a
DeprecationWarning; support will be removed in a future version.
"""
from __future__ import annotations

import datetime
import inspect
import logging
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import awkward as ak

from lmetk.data.data_objects.data import (
    NamedTransformInput,
    SplitDatasetContainer,
)
from lmetk.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from lmetk.data.data_objects.slice_info import SliceInfo
from lmetk.transforms.base.integrity import run_checks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Integrity check type aliases
# ---------------------------------------------------------------------------

# Before-checks: fn(split_chunks) -> None  where split_chunks is
#     Dict[str, List[NamedTransformInput]] — one split's worth of chunks.
BeforeCheckFn = Callable[[Dict[str, List[NamedTransformInput]]], None]

# After-checks: fn(assign_to_slice) -> None  where assign_to_slice is
#     Dict[str, Any] — only the assign_to keys from transformed_data.
AfterCheckFn = Callable[[Dict[str, Any]], None]

# A check triple: (name, fn, policy).  policy ∈ {"fail", "warn", "allow"}.
BeforeCheck = Tuple[str, BeforeCheckFn, str]
AfterCheck = Tuple[str, AfterCheckFn, str]


# ---------------------------------------------------------------------------
# TransformContext
# ---------------------------------------------------------------------------

@dataclass
class TransformContext:
    """
    Shared state threaded through the transform pipeline.

    Inputs (set by TransformStrategy.apply before the pipeline runs)
    ---------------------------------------------------------------
    data, transform_instance, apply_to_keys, assign_to_keys,
    assign_to_keys_map, fit_on_split, fit_on_key, transform_on_keys,
    strategy, postprocess_fn, slice_info,
    integrity_checks_before, integrity_checks_after.

    Chunk builders (set by RegisterChunkBuilderStep)
    ------------------------------------------------
    build_fit_chunks          callable → List[NamedTransformInput]
    build_transform_chunks    callable(split) → List[NamedTransformInput]

    Chunks are NOT stored on context. Each consumer calls the builder,
    holds a local reference, and lets it go out of scope when done.

    Mutable pipeline state (set by steps in order)
    -----------------------------------------------
    transformed_data               set by CopyStep
    available_splits               set by CopyStep
    transformed_results_for_new_key  set by CopyStep, filled by PostprocessStep
    _raw_transformed_by_split      set by TransformStep, read by PostprocessStep
    log                            updated by TransformStep

    Integrity run state (set by integrity steps; read by RecordManifestStep)
    -------------------------------------------------------------------------
    integrity_before_ran, integrity_before_check_names
    integrity_after_ran,  integrity_after_check_names
    """

    # --- Inputs ---
    data: SplitDatasetContainer
    transform_instance: Any
    apply_to_keys: List[str]
    assign_to_keys: List[str]
    assign_to_keys_map: List[str]
    fit_on_split: Optional[str] = None
    fit_on_key: Optional[str] = None
    transform_on_keys: Optional[List[str]] = None

    # --- Infrastructure ---
    strategy: Optional[Any] = None
    postprocess_fn: Optional[Callable[..., Any]] = None
    slice_info: Optional[SliceInfo] = None
    step_id: Optional[str] = None

    # --- Integrity check lists ---
    integrity_checks_before: Optional[List[BeforeCheck]] = None
    integrity_checks_after: Optional[List[AfterCheck]] = None
    # Legacy: single list for backward compat when old strategy runs (IntegrityCheckStep).
    integrity_checks: Optional[List[Tuple[str, Callable[[], None], str]]] = None

    # --- Chunk builders (set by RegisterChunkBuilderStep) ---
    # Callables, not data. Consumers call these per-split so chunks are streamed.
    build_fit_chunks: Optional[Callable[[], List[NamedTransformInput]]] = None
    build_transform_chunks: Optional[Callable[[str], List[NamedTransformInput]]] = None

    # --- Mutable pipeline state ---
    log: Dict[str, Any] = field(default_factory=dict)
    transformed_data: Optional[SplitDatasetContainer] = None
    available_splits: Optional[List[str]] = None
    transformed_results_for_new_key: Optional[Dict[str, Dict[str, List[Any]]]] = None
    _raw_transformed_by_split: Dict[str, Any] = field(default_factory=dict)

    # --- Integrity run state ---
    integrity_before_ran: bool = False
    integrity_before_check_names: List[str] = field(default_factory=list)
    integrity_after_ran: bool = False
    integrity_after_check_names: List[str] = field(default_factory=list)

    # --- Observability ---
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cache_status: Optional[str] = None
    error: Optional[BaseException] = None


# ---------------------------------------------------------------------------
# PipelineStep protocol
# ---------------------------------------------------------------------------

class PipelineStep(Protocol):
    def run(self, context: TransformContext) -> TransformContext:
        ...


# ---------------------------------------------------------------------------
# Hook registry
# ---------------------------------------------------------------------------

HookCallback = Callable[[str, TransformContext], None]
_HOOK_REGISTRY: Dict[str, List[HookCallback]] = {
    "on_pipeline_start": [],
    "before_transform": [],
    "after_transform": [],
    "on_transform_error": [],
    "on_pipeline_end": [],
}


def register_hook(event: str, callback: HookCallback) -> None:
    if event not in _HOOK_REGISTRY:
        _HOOK_REGISTRY[event] = []
    _HOOK_REGISTRY[event].append(callback)


def clear_hooks(event: Optional[str] = None) -> None:
    if event is None:
        for k in _HOOK_REGISTRY:
            _HOOK_REGISTRY[k] = []
    elif event in _HOOK_REGISTRY:
        _HOOK_REGISTRY[event] = []


def _emit(event: str, context: TransformContext) -> None:
    for cb in _HOOK_REGISTRY.get(event, []):
        try:
            cb(event, context)
        except Exception:
            pass  # hooks must never break the pipeline


def _register_default_hooks() -> None:
    from lmetk.transforms.base import hooks
    hooks.register_defaults()


_register_default_hooks()


# ---------------------------------------------------------------------------
# Integrity check runner
# ---------------------------------------------------------------------------

def _run_scoped_checks(
    checks: List[Tuple[str, Callable, str]],
    data_slice: Any,
) -> List[str]:
    """
    Run (name, fn, policy) checks, passing data_slice as the sole argument.

    Supports legacy no-arg functions via introspection; emits DeprecationWarning.
    Returns the names of checks that completed (passed or warn-policy failures).
    Raises RuntimeError immediately on policy="fail" failures.
    """
    ran: List[str] = []
    for name, fn, policy in checks:
        try:
            sig = inspect.signature(fn)
            if len(sig.parameters) == 0:
                warnings.warn(
                    f"Integrity check '{name}' uses a no-arg signature. "
                    "Pass the data slice as the first argument. "
                    "No-arg checks will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                fn()
            else:
                fn(data_slice)
            ran.append(name)
        except Exception as exc:
            if policy == "fail":
                raise RuntimeError(
                    f"Integrity check '{name}' failed: {exc}"
                ) from exc
            elif policy == "warn":
                logger.warning(
                    "Integrity check '%s' failed (warn only): %s", name, exc
                )
                ran.append(name)
            # policy == "allow": swallow silently

    return ran


# ---------------------------------------------------------------------------
# Default integrity check functions
# ---------------------------------------------------------------------------

def _default_before_check_nans(
    split_chunks: Dict[str, List[NamedTransformInput]],
) -> None:
    """Default before-check: no NaN values in any chunk of this split."""
    from lmetk.data.data_objects.utils import check_for_nans
    for split, chunks in split_chunks.items():
        for chunk in chunks:
            check_for_nans(
                list(chunk.values()), list(chunk.keys()), raise_on_error=True
            )


def _default_before_check_lengths(
    split_chunks: Dict[str, List[NamedTransformInput]],
) -> None:
    """Default before-check: consistent array lengths within each chunk."""
    from lmetk.data.data_objects.utils import check_length_consistency
    for split, chunks in split_chunks.items():
        for chunk in chunks:
            check_length_consistency(
                list(chunk.values()), list(chunk.keys()), raise_on_error=True
            )


def _default_after_check_structure(
    assign_to_slice: Dict[str, Any],
) -> None:
    """Default after-check: every assign_to key maps to a split-keyed dict."""
    for key, split_data in assign_to_slice.items():
        if not isinstance(split_data, Mapping):
            raise TypeError(
                f"Output key '{key}' must be a split-keyed mapping, "
                f"got {type(split_data).__name__}."
            )


def _default_after_check_splits(
    assign_to_slice: Dict[str, Any],
    original_splits: frozenset,
) -> None:
    """Default after-check: split set is unchanged after transformation."""
    for key, split_data in assign_to_slice.items():
        if isinstance(split_data, Mapping):
            result_splits = frozenset(split_data.keys())
            if result_splits != original_splits:
                raise KeyError(
                    f"Output key '{key}': split set changed. "
                    f"Expected {set(original_splits)}, got {set(result_splits)}."
                )


def build_default_before_checks(policy: str = "fail") -> List[BeforeCheck]:
    """
    Return the default before-checks: NaN detection + length consistency.

    Called by TransformStrategy.apply() when validate_output=True and no
    custom before-checks are provided.
    """
    return [
        ("check_no_nans", _default_before_check_nans, policy),
        ("check_length_consistency", _default_before_check_lengths, policy),
    ]


def build_default_after_checks(
    original_splits: frozenset,
    policy: str = "fail",
) -> List[AfterCheck]:
    """
    Return the default after-checks: structure validity + split preservation.

    Requires original_splits so the split-preservation check can compare.
    Called by TransformStrategy.apply() when validate_output=True and no
    custom after-checks are provided.
    """
    def _check_splits(assign_to_slice: Dict[str, Any]) -> None:
        _default_after_check_splits(assign_to_slice, original_splits)

    return [
        ("check_output_structure", _default_after_check_structure, policy),
        ("check_split_preservation", _check_splits, policy),
    ]


# ---------------------------------------------------------------------------
# Step 1: CopyStep
# ---------------------------------------------------------------------------

class CopyStep:

    def run(self, context: TransformContext) -> TransformContext:
        data = context.data
        apply_to_keys = context.apply_to_keys
        assign_to_keys = context.assign_to_keys or []

        if not apply_to_keys:
            raise ValueError("apply_to_keys must not be empty.")

        apply_key = apply_to_keys[0]
        if apply_key not in data:
            raise KeyError(
                f"apply_to key '{apply_key}' not found in data. "
                f"Available: {sorted(str(k) for k in data.keys())}"
            )
        if not isinstance(data[apply_key], Mapping):
            raise TypeError(f"data['{apply_key}'] must be a split-keyed mapping.")

        transformed_data = data.copy(deep=False)
        for key in set(list(apply_to_keys) + list(assign_to_keys)):
            if key in data:
                first_leaf = self._first_leaf(data[key])
                if isinstance(first_leaf, ak.Array):
                    # ak.Array does not support item assignment — shallow copy is safe
                    # and avoids an unnecessary memory allocation.
                    transformed_data[key] = data[key].copy(deep=False)
                else:
                    # np.ndarray is mutable in place — deep copy to protect original data.
                    transformed_data[key] = data[key].copy(deep=True)

        available_splits = list(data[apply_key].keys())
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
        Unwrap nested dicts and lists to find the first concrete array.

        data[key] structure is typically: {split: [unit_array, unit_array, ...]}.
        We need to reach an actual array to determine its type.
        """
        while isinstance(val, (Mapping, list)):
            val = next(iter(val.values())) if isinstance(val, Mapping) else val[0]
        return val


# ---------------------------------------------------------------------------
# Step 2: RegisterChunkBuilderStep
# ---------------------------------------------------------------------------

class RegisterChunkBuilderStep:
    """
    Register chunk-building callables on context. No chunks are built here.

    Stores two callables:
        context.build_fit_chunks()         → List[NamedTransformInput]
        context.build_transform_chunks(split) → List[NamedTransformInput]

    Each callable builds chunks on demand and returns them. The caller holds
    a local reference for the duration of its work; when the local goes out
    of scope the chunks are eligible for GC. This means at most one split's
    worth of chunk wrappers exists in memory at any time regardless of the
    number of splits or units — critical for PHM datasets.

    Chunk wrappers hold *references* to the underlying arrays in context.data;
    no array data is copied.

    Sets:  context.build_fit_chunks, context.build_transform_chunks.
    Reads: context.data, context.apply_to_keys, context.fit_on_split,
           context.fit_on_key, context.strategy.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if context.strategy is None:
            return context  # hand-built context (e.g. unit test); steps fall back inline

        strategy = context.strategy
        data = context.data
        is_multi_unit = data.is_multi_unit
        apply_to_keys = context.apply_to_keys
        fit_on_split = context.fit_on_split
        fit_on_key = context.fit_on_key

        def build_transform_chunks(split: str) -> List[NamedTransformInput]:
            target_data = {key: data[key] for key in apply_to_keys}
            return strategy._prepare_chunks_for_split(target_data, split, is_multi_unit)

        def build_fit_chunks() -> List[NamedTransformInput]:
            if not fit_on_split or not fit_on_key:
                return []
            fit_data = {fit_on_key: data[fit_on_key]}
            return strategy._prepare_chunks_for_split(
                fit_data, fit_on_split, is_multi_unit
            )

        context.build_transform_chunks = build_transform_chunks
        context.build_fit_chunks = build_fit_chunks
        return context


# ---------------------------------------------------------------------------
# Step 2.5: IntegrityCheckStep (legacy — for backward compat with old strategy)
# ---------------------------------------------------------------------------

class IntegrityCheckStep:
    """
    Legacy step: run optional no-arg integrity checks (context.integrity_checks).
    Used when the old strategy runs (CopyStep → IntegrityCheckStep → FitStep → ...).
    No-op if context.integrity_checks is None. New strategy uses IntegrityCheckBeforeStep
    and IntegrityCheckAfterStep instead.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if not context.integrity_checks:
            return context
        run_checks(context.integrity_checks)
        return context


# ---------------------------------------------------------------------------
# Step 3: IntegrityCheckBeforeStep
# ---------------------------------------------------------------------------

class IntegrityCheckBeforeStep:
    """
    Run before-checks over apply_to chunks, one split at a time.

    Calls context.build_transform_chunks(split) for each available split,
    passes the resulting {split: chunks} dict to each check function, then
    discards the chunks before moving to the next split. Peak memory is one
    split's chunks, not all splits simultaneously.

    Check functions receive Dict[str, List[NamedTransformInput]] and cannot
    accidentally access the full container.

    Default checks (NaN + length consistency) are injected by
    TransformStrategy.apply() when validate_output=True and no custom
    before-checks are provided.

    Sets:  context.integrity_before_ran, context.integrity_before_check_names.
    Reads: context.build_transform_chunks, context.available_splits,
           context.integrity_checks_before.
    """

    def run(self, context: TransformContext) -> TransformContext:
        checks = context.integrity_checks_before
        if not checks:
            return context

        build = context.build_transform_chunks
        if build is None or context.available_splits is None:
            return context  # RegisterChunkBuilderStep did not run

        ran: List[str] = []
        for split in context.available_splits:
            # Build one split, check, discard.
            split_chunks = {split: build(split)}
            ran_for_split = _run_scoped_checks(checks, split_chunks)
            # Accumulate unique names (same checks run for every split).
            for name in ran_for_split:
                if name not in ran:
                    ran.append(name)
            # split_chunks goes out of scope here → GC before next split.

        context.integrity_before_check_names = ran
        context.integrity_before_ran = True
        return context


# ---------------------------------------------------------------------------
# Step 4: FitStep
# ---------------------------------------------------------------------------

class FitStep:
    """
    Fit the transform on context.build_fit_chunks() when fit_on_split/key are set.

    Calls the builder once, fits, then the local reference goes out of scope.
    Falls back to building inline if the builder was not registered (test contexts).

    Reads: context.build_fit_chunks (from RegisterChunkBuilderStep),
           context.fit_on_split, context.fit_on_key, context.transform_instance.
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

        # Get chunks: prefer registered builder, fall back for test contexts.
        build = context.build_fit_chunks
        if build is not None:
            fit_chunks = build()  # local reference only
        elif context.strategy is not None:
            logger.debug(
                "FitStep: build_fit_chunks not registered "
                "(RegisterChunkBuilderStep may not have run). Building inline."
            )
            fit_data = {context.fit_on_key: context.data[context.fit_on_key]}
            fit_chunks = context.strategy._prepare_chunks_for_split(
                fit_data, context.fit_on_split, context.data.is_multi_unit
            )
        else:
            return context  # nothing we can do

        if not fit_chunks:
            return context

        metadata = {
            "mode": context.fit_on_split,
            "apply_to_keys": context.fit_on_key,
            "assign_to_keys": context.assign_to_keys,
            "assign_to_map": context.assign_to_keys_map,
        }

        if context.data.is_multi_unit:
            context.transform_instance.fit_multi_source(fit_chunks, metadata=metadata)
        else:
            context.transform_instance.fit_data(fit_chunks[0], metadata=metadata)

        # fit_chunks goes out of scope here → GC.
        return context


# ---------------------------------------------------------------------------
# Step 5: TransformStep
# ---------------------------------------------------------------------------

class TransformStep:
    """
    Apply the transform to each split, streaming one split at a time.

    For each split: calls build_transform_chunks(split), transforms, stores
    the raw result, then the chunks local goes out of scope before the next
    split is built. Peak memory is one split's chunks at a time.

    Falls back to building inline if the builder was not registered.

    Reads: context.build_transform_chunks, context.available_splits,
           context.transform_instance, context.data.
    Sets:  context._raw_transformed_by_split, context.log.
    """

    def run(self, context: TransformContext) -> TransformContext:
        if context.available_splits is None:
            return context

        build = context.build_transform_chunks

        for split in context.available_splits:
            # --- Get chunks for this split only ---
            if build is not None:
                chunks = build(split)  # local; GC'd after this iteration
            elif context.strategy is not None:
                logger.debug(
                    "TransformStep: build_transform_chunks not registered. "
                    "Building inline for split '%s'.", split
                )
                target_data = {key: context.data[key] for key in context.apply_to_keys}
                chunks = context.strategy._prepare_chunks_for_split(
                    target_data, split, context.data.is_multi_unit
                )
            else:
                continue

            metadata = {
                "mode": split,
                "apply_to_keys": context.apply_to_keys,
                "assign_to_keys": context.assign_to_keys,
                "assign_to_map": context.assign_to_keys_map,
            }

            if context.data.is_multi_unit:
                transformed_chunks, log = (
                    context.transform_instance.transform_multi_source(
                        chunks, metadata=metadata
                    )
                )
                context.log[split] = log
            else:
                transformed_chunks = [
                    context.transform_instance.transform_data(
                        chunks[0], metadata=metadata
                    )
                ]

            context._raw_transformed_by_split[split] = (transformed_chunks, metadata)
            # chunks goes out of scope here → GC before next split.

        return context


# ---------------------------------------------------------------------------
# Step 6: PostprocessStep
# ---------------------------------------------------------------------------

class PostprocessStep:
    """
    Normalise raw transform output and fill transformed_results_for_new_key.

    Reads:  context._raw_transformed_by_split, context.postprocess_fn,
            context.assign_to_keys, context.assign_to_keys_map.
    Sets:   context.transformed_results_for_new_key (filled).
    """

    def run(self, context: TransformContext) -> TransformContext:
        if (
            context.postprocess_fn is None
            or context.transformed_results_for_new_key is None
        ):
            return context

        for split, (transformed_chunks, metadata) in (
            context._raw_transformed_by_split.items()
        ):
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


# ---------------------------------------------------------------------------
# Step 7: MergeStep
# ---------------------------------------------------------------------------

class MergeStep:
    """
    Merge per-split transformed results into context.transformed_data.

    Soft diagnostic warnings (type change, unit-count change) that were
    previously in DataTransform._validate_forward_output now live here,
    where before/after data is locally available without extra arguments.
    Hard structural failures are handled by IntegrityCheckAfterStep.

    Reads:  context.transformed_results_for_new_key, context.transformed_data,
            context.available_splits, context.strategy.
    Mutates: context.transformed_data (in place via strategy._merge_transformed_data).
    """

    def run(self, context: TransformContext) -> TransformContext:
        if (
            context.strategy is None
            or context.transformed_data is None
            or context.available_splits is None
            or context.transformed_results_for_new_key is None
        ):
            return context

        self._warn_on_changes(context)

        context.strategy._merge_transformed_data(
            base_data=context.transformed_data,
            transformed_results=context.transformed_results_for_new_key,
            available_splits=context.available_splits,
        )
        return context

    @staticmethod
    def _warn_on_changes(context: TransformContext) -> None:
        """Emit diagnostic warnings for type or unit-count changes."""
        transform_name = type(context.transform_instance).__name__
        for assign_to_key in context.transformed_results_for_new_key:
            if assign_to_key not in context.data:
                continue  # new key — no "before" to compare
            for split in context.available_splits:
                before = context.data[assign_to_key].get(split)
                after_list = context.transformed_results_for_new_key[
                    assign_to_key
                ].get(split)
                if before is None or after_list is None:
                    continue
                if not isinstance(after_list, type(before)):
                    logger.warning(
                        "Transform '%s': type of '%s/%s' changed from %s to %s.",
                        transform_name, assign_to_key, split,
                        type(before).__name__, type(after_list).__name__,
                    )
                if isinstance(before, list) and isinstance(after_list, list):
                    if len(before) != len(after_list):
                        logger.warning(
                            "Transform '%s': unit count of '%s/%s' changed "
                            "from %d to %d.",
                            transform_name, assign_to_key, split,
                            len(before), len(after_list),
                        )


# ---------------------------------------------------------------------------
# Step 8: IntegrityCheckAfterStep
# ---------------------------------------------------------------------------

class IntegrityCheckAfterStep:
    """
    Run after-checks over the assign_to output in context.transformed_data.

    Builds a slice containing only the assign_to keys and passes it to each
    check function — checks cannot accidentally inspect unrelated keys.

    Default checks (structure + split preservation) are injected by
    TransformStrategy.apply() when validate_output=True and no custom
    after-checks are provided.

    Sets:  context.integrity_after_ran, context.integrity_after_check_names.
    Reads: context.transformed_data, context.assign_to_keys,
           context.integrity_checks_after.
    """

    def run(self, context: TransformContext) -> TransformContext:
        checks = context.integrity_checks_after
        if not checks or context.transformed_data is None:
            return context

        # Ensure all assign_to keys are present (same contract as former _validate_forward_output).
        for key in context.assign_to_keys:
            if key not in context.transformed_data:
                raise KeyError(
                    f"Transformed data from strategy does not contain '{key}' key at the top level."
                )

        # Build the assign_to slice: only keys this transform produced.
        assign_to_slice: Dict[str, Any] = {
            key: context.transformed_data[key]
            for key in context.assign_to_keys
            if key in context.transformed_data
        }

        ran = _run_scoped_checks(checks, assign_to_slice)
        context.integrity_after_check_names = ran
        context.integrity_after_ran = True
        return context


# ---------------------------------------------------------------------------
# Step 9: RecordManifestStep
# ---------------------------------------------------------------------------

def _get_producer_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("lmetk")
    except Exception:
        return "0.0.0"


class RecordManifestStep:
    """
    Append a transform metadata entry to the container's manifest.

    Payload now includes integrity_before/after fields recording that the
    integrity steps ran and which checks were executed.

    Reads:  context.transformed_data (manifest attribute), context.*.
    Mutates: manifest (append only).
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
                if context.transform_instance else None
            ),
            "apply_to_keys": context.apply_to_keys,
            "assign_to_keys": context.assign_to_keys,
            "fit_on_split": context.fit_on_split,
            "fit_on_key": context.fit_on_key,
            "log_splits": list(context.log.keys()) if context.log else [],
            # Integrity provenance (Phase 3 of the plan)
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
                    if context.slice_info.unit_ids else None
                ),
                "n_cycles": (
                    len(context.slice_info.cycle_ids)
                    if context.slice_info.cycle_ids else None
                ),
            }

        split_str = (
            ",".join(context.available_splits)
            if context.available_splits else None
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
