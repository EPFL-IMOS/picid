"""
Phase 3: Composable pipeline steps and lifecycle hooks.

The pipeline runs in a fixed order (see TransformStrategy.apply): CopyStep → IntegrityCheckStep
→ FitStep → TransformStep → PostprocessStep → MergeStep → RecordManifestStep. Each step reads
and/or mutates TransformContext; later steps depend on fields set by earlier steps.

- TransformContext: holds input data, transform_instance, metadata (apply_to_keys, fit_on_split, etc.),
  and mutable state that steps fill (transformed_data, available_splits, _raw_transformed_by_split, etc.).
- PipelineStep: protocol run(context) -> context.
- Hook registry and events: on_pipeline_start, before_transform, after_transform, on_transform_error,
  on_pipeline_end. When on_transform_error is emitted, context.error is set to the exception.
  Default hook implementations live in lmetk.transforms.base.hooks. See docs/transforms/HOOKS_EXAMPLES.md.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from lmetk.data.data_objects.data import SplitDatasetContainer
from lmetk.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from lmetk.data.data_objects.slice_info import SliceInfo
from lmetk.transforms.base.integrity import run_checks


@dataclass
class TransformContext:
    """
    Shared context passed through the transform pipeline. Steps read and update this object in order.

    Inputs (set when the context is built, e.g. by TransformStrategy.apply):
    - data, transform_instance, apply_to_keys, assign_to_keys, assign_to_keys_map, fit_on_split,
      fit_on_key, transform_on_keys, strategy, postprocess_fn, slice_info, integrity_checks.

    Mutable state (set by steps; later steps depend on these):
    - transformed_data, available_splits, transformed_results_for_new_key: set by CopyStep; later
      steps read or extend them.
    - _raw_transformed_by_split: set by TransformStep; read by PostprocessStep.
    - log: updated by TransformStep (per-split logs from transform_multi_source).

    Attributes
    ----------
    data : SplitDatasetContainer
        Original input data (read-only for hooks; steps may read).
    transform_instance : Any
        The transform being applied.
    apply_to_keys : List[str]
        Keys to apply the transform to.
    assign_to_keys : List[str]
        Keys to assign results to.
    assign_to_keys_map : List[str]
        Map from assign_to key to chunk key (e.g. assign_to "features" -> chunk key "features").
    fit_on_split : Optional[str]
        Split used for fitting (e.g. "train"), or None.
    fit_on_key : Optional[str]
        Key used for fitting, or None.
    transform_on_keys : Optional[List[str]]
        If set, only these splits are transformed.
    log : Dict[str, Any]
        Transform log (e.g. per-split logs from transform_multi_source).
    transformed_data : Optional[SplitDatasetContainer]
        Set by CopyStep; holds the copy that will receive merged results. Read by MergeStep, RecordManifestStep.
    available_splits : Optional[List[str]]
        Set by CopyStep; list of split names to transform. Read by TransformStep, MergeStep.
    transformed_results_for_new_key : Optional[Dict]
        Structure set by CopyStep (defaultdict(dict)); filled by PostprocessStep; read by MergeStep.
    """

    data: SplitDatasetContainer
    transform_instance: Any
    apply_to_keys: List[str]
    assign_to_keys: List[str]
    assign_to_keys_map: List[str]
    fit_on_split: Optional[str] = None
    fit_on_key: Optional[str] = None
    transform_on_keys: Optional[List[str]] = None
    log: Dict[str, Any] = field(default_factory=dict)

    # Mutable state (set by steps)
    transformed_data: Optional[SplitDatasetContainer] = None
    available_splits: Optional[List[str]] = None
    transformed_results_for_new_key: Optional[Dict[str, Dict[str, List[Any]]]] = None
    # Raw transform output per split: (transformed_chunks, metadata); used by TransformStep -> PostprocessStep
    _raw_transformed_by_split: Dict[str, Any] = field(default_factory=dict)

    # Optional: for hooks and observability
    step_id: Optional[str] = None
    cache_status: Optional[str] = None
    # Timing: set by TransformStrategy.apply() for manifest (started_at/finished_at as epoch seconds)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Set by TransformStrategy.apply() when building context (strategy=self). Steps use it to call
    # _prepare_chunks_for_split and _merge_transformed_data. In production it is always an instance
    # of TransformStrategy; it is None only when context is built by hand (e.g. in tests).
    strategy: Optional[Any] = None
    # Set by TransformStrategy.apply() when building context (postprocess_fn=postprocess_transformed_data).
    # Used by PostprocessStep to normalize raw transform output into list of SimpleReturnObject. None if context built by hand.
    postprocess_fn: Optional[Callable[..., Any]] = None
    # Slice-awareness (Phase 4.2): copied from data.slice_info when building context; read-only for transforms
    slice_info: Optional[SliceInfo] = None
    # Integrity (Phase 4.3): optional list of (name, check_fn, policy); IntegrityCheckStep runs them
    integrity_checks: Optional[List[Tuple[str, Callable[[], None], str]]] = None
    # Set by strategy when emitting on_transform_error; hooks can read for debugging
    error: Optional[BaseException] = None


# -----------------------------------------------------------------------------
# PipelineStep protocol
# -----------------------------------------------------------------------------


class PipelineStep(Protocol):
    """
    Protocol for a pipeline step. Each step receives the same TransformContext and returns it
    (possibly mutated). Steps are invoked in sequence by TransformStrategy.apply; order matters
    because later steps depend on fields set by earlier steps.
    """

    def run(self, context: TransformContext) -> TransformContext:
        """Execute the step; may mutate context. Returns context for chaining."""
        ...


# -----------------------------------------------------------------------------
# Hook registry and events (Phase 3.3)
# -----------------------------------------------------------------------------

HookCallback = Callable[[str, TransformContext], None]
_HOOK_REGISTRY: Dict[str, List[HookCallback]] = {
    "on_pipeline_start": [],
    "before_transform": [],
    "after_transform": [],
    "on_transform_error": [],
    "on_pipeline_end": [],
}


def register_hook(event: str, callback: HookCallback) -> None:
    """Register a callback for an event. Event is one of the keys in _HOOK_REGISTRY."""
    if event not in _HOOK_REGISTRY:
        _HOOK_REGISTRY[event] = []
    _HOOK_REGISTRY[event].append(callback)


def clear_hooks(event: Optional[str] = None) -> None:
    """Clear hooks for one event or all events."""
    if event is None:
        for k in _HOOK_REGISTRY:
            _HOOK_REGISTRY[k] = []
    elif event in _HOOK_REGISTRY:
        _HOOK_REGISTRY[event] = []


def _emit(event: str, context: TransformContext) -> None:
    """Emit an event; callbacks receive (event, context). Context is read-only for hooks."""
    for cb in _HOOK_REGISTRY.get(event, []):
        try:
            cb(event, context)
        except Exception:
            pass  # Hooks must not break the pipeline


# Default hooks (implementations in hooks.py) are registered when the pipeline is used
def _register_default_hooks() -> None:
    from lmetk.transforms.base import hooks
    hooks.register_defaults()


_register_default_hooks()


# -----------------------------------------------------------------------------
# Concrete pipeline steps (Phase 3.2)
# -----------------------------------------------------------------------------

from collections import defaultdict

import awkward as ak


class CopyStep:
    """
    Step 1: Copy container and establish split list. No prior step; required for all later steps.

    What it does:
    - Shallow-copies context.data into a new container (deep copy for non-awkward keys to modify).
    - Validates apply_to_keys (non-empty, apply key exists in data, structure is split-keyed).
    - Applies transform_on_keys filter: available_splits = data splits ∩ transform_on_keys (raises if empty).
    - Sets context.transformed_data, context.available_splits, context.transformed_results_for_new_key
      (defaultdict(dict)), and clears context._raw_transformed_by_split.

    Depends on: context.data, context.apply_to_keys, context.transform_on_keys (optional).
    Affects: FitStep (uses strategy + data; fit does not need transformed_data), TransformStep
    (needs transformed_data, available_splits), PostprocessStep (needs transformed_results_for_new_key
    container), MergeStep (needs all three), RecordManifestStep (needs transformed_data).
    """

    def run(self, context: TransformContext) -> TransformContext:
        data = context.data
        apply_to_keys = context.apply_to_keys
        assign_to_keys = context.assign_to_keys or []

        transformed_data = data.copy(deep=False)
        keys_to_modify = list(apply_to_keys) + list(assign_to_keys)
        for key in keys_to_modify:
            if key in data:
                if data[key].get_instance_cls() is not ak.Array:
                    transformed_data[key] = data[key].copy(deep=True)
                else:
                    transformed_data[key] = data[key].copy(deep=False)

        if not apply_to_keys:
            raise ValueError(
                "No 'apply_to_keys' provided. Provide at least one key to apply the transform."
            )
        apply_key = apply_to_keys[0]
        if apply_key not in data:
            available_keys = ", ".join(sorted(map(str, list(data.keys()))))
            raise KeyError(
                f"Apply-to key '{apply_key}' not found in data. "
                f"Available keys: [{available_keys}]"
            )
        if not isinstance(data[apply_key], Mapping):
            raise TypeError(
                f"Unexpected structure for data['{apply_key}']; expected a mapping with split keys."
            )
        available_splits = list(data[apply_key].keys())
        if context.transform_on_keys is not None:
            available_splits = list(set(available_splits) & set(context.transform_on_keys))
            if not available_splits:
                data_splits = sorted(data[apply_key].keys())
                requested = sorted(context.transform_on_keys)
                transform_name = getattr(
                    context.transform_instance, "__name__", type(context.transform_instance).__name__
                )
                raise ValueError(
                    f"Transform {transform_name!r}: no splits left after applying transform_on_keys. "
                    f"Data has splits: {data_splits}. Requested transform_on_keys: {requested}. "
                    "Fix: set metadata.transform_on_keys to include at least one split present in the data "
                    f"(e.g. one of {data_splits}), or set transform_on_keys to None to run on all splits."
                )

        context.transformed_data = transformed_data
        context.available_splits = available_splits
        context.transformed_results_for_new_key = defaultdict(dict)
        context._raw_transformed_by_split = {}
        return context


class IntegrityCheckStep:
    """
    Step 1.5 (Phase 4.3): Run optional integrity checks. No-op if context.integrity_checks is None.

    What it does:
    - Runs each (name, check_fn, policy) in context.integrity_checks via run_checks (fail / warn / allow).
    - Does not read or set pipeline state; used to validate context or data after CopyStep.

    Depends on: context.integrity_checks (set when context is built). No dependency on other steps.
    Affects: None (checks must not mutate context for later steps).
    """

    def run(self, context: TransformContext) -> TransformContext:
        if not context.integrity_checks:
            return context
        run_checks(context.integrity_checks)
        return context


class FitStep:
    """
    Step 2: Fit the transform on the specified split when fit_on_split and fit_on_key are set.

    What it does:
    - If transform has requires_fit=True and fit_on_split or fit_on_key is missing, raises ValueError.
    - If either is missing and transform does not require fit, returns without fitting.
    - Otherwise uses context.strategy._prepare_chunks_for_split to get chunks for the fit split,
      then calls transform_instance.fit_multi_source or fit_data. Does not write to transformed_data.

    Depends on: context.strategy (from TransformStrategy.apply), context.data, context.fit_on_split,
    context.fit_on_key, context.transform_instance. Does not require CopyStep outputs (only strategy).
    Affects: TransformStep (transform_instance is fitted and used in transform_data / transform_multi_source).
    """

    def run(self, context: TransformContext) -> TransformContext:
        requires_fit = getattr(context.transform_instance, "requires_fit", False)
        if not context.fit_on_split or not context.fit_on_key:
            if requires_fit:
                transform_name = getattr(
                    context.transform_instance, "__name__", type(context.transform_instance).__name__
                )
                raise ValueError(
                    f"Transform {transform_name!r} requires fitting but fit_on_split and/or fit_on_key are not set. "
                    "Set metadata 'fit_on' (e.g. 'train') and optionally 'fit_on_key' (defaults to apply_to)."
                )
            return context
        strategy = context.strategy  # TransformStrategy from strategy.apply(); None if context built by hand
        if strategy is None:
            return context
        data = context.data
        is_multi_unit = data.is_multi_unit
        fit_chunks = strategy._prepare_chunks_for_split(
            {context.fit_on_key: data[context.fit_on_key]},
            context.fit_on_split,
            is_multi_unit,
        )
        metadata = {
            "mode": context.fit_on_split,
            "apply_to_keys": context.fit_on_key,
            "assign_to_keys": context.assign_to_keys,
            "assign_to_map": context.assign_to_keys_map,
        }
        if is_multi_unit:
            context.transform_instance.fit_multi_source(fit_chunks, metadata=metadata)
        else:
            context.transform_instance.fit_data(fit_chunks[0], metadata=metadata)
        return context


class TransformStep:
    """
    Step 3: Run the transform on each split and store raw output per split.

    What it does:
    - For each split in context.available_splits, prepares chunks via strategy._prepare_chunks_for_split,
      calls transform_instance.transform_multi_source (multi-unit) or transform_data (single-unit),
      and stores (transformed_chunks, metadata) in context._raw_transformed_by_split[split].
    - Updates context.log[split] when transform_multi_source returns a log. Does not merge into
      transformed_data; that is done by MergeStep after PostprocessStep.

    Depends on: CopyStep (context.transformed_data, context.available_splits), FitStep (transform
    must be fitted if it requires fitting), context.strategy, context.data, context.apply_to_keys.
    Affects: PostprocessStep (reads _raw_transformed_by_split), MergeStep indirectly (via PostprocessStep
    filling transformed_results_for_new_key).
    """

    def run(self, context: TransformContext) -> TransformContext:
        strategy = context.strategy
        # Skip when prerequisites are missing:
        # - strategy is None: context was built by hand (e.g. tests) and we cannot call _prepare_chunks_for_split.
        # - transformed_data or available_splits is None: only CopyStep sets these (context defaults are None).
        #   So they are None only if CopyStep was never run (e.g. TransformStep run in isolation in tests, or
        #   steps invoked out of order). In the full pipeline, CopyStep always runs first, so both are set.
        if strategy is None:
            return context
        if context.transformed_data is None or context.available_splits is None:
            return context
        target_data = {key: context.data[key] for key in context.apply_to_keys}
        is_multi_unit = context.data.is_multi_unit
        for split in context.available_splits:
            chunks_to_transform = strategy._prepare_chunks_for_split(
                target_data, split, is_multi_unit
            )
            metadata = {
                "mode": split,
                "apply_to_keys": context.apply_to_keys,
                "assign_to_keys": context.assign_to_keys,
                "assign_to_map": context.assign_to_keys_map,
            }
            if is_multi_unit:
                transformed_chunks, log = context.transform_instance.transform_multi_source(
                    chunks_to_transform, metadata=metadata
                )
                context.log[split] = log
            else:
                transformed_chunks = [
                    context.transform_instance.transform_data(
                        chunks_to_transform[0], metadata=metadata
                    )
                ]
            context._raw_transformed_by_split[split] = (transformed_chunks, metadata)
        return context


class PostprocessStep:
    """
    Step 4: Normalize raw transform output and fill transformed_results_for_new_key per split.

    What it does:
    - For each split, reads (transformed_chunks, metadata) from context._raw_transformed_by_split,
      calls context.postprocess_fn(transformed_chunks, metadata) to get a list of objects with
      assign_to_map keys, then writes context.transformed_results_for_new_key[assign_to_key][split]
      from each chunk's assign_to_map value. Fails if a chunk is missing the expected key.

    Depends on: CopyStep (transformed_results_for_new_key structure), TransformStep
    (_raw_transformed_by_split), context.postprocess_fn (set when context is built).
    Affects: MergeStep (reads transformed_results_for_new_key to merge into transformed_data).
    """

    def run(self, context: TransformContext) -> TransformContext:
        # Skip when prerequisites are missing:
        # - postprocess_fn: set only in TransformStrategy.apply() when building the context
        #   (postprocess_fn=postprocess_transformed_data). None only if context was built by hand (e.g. tests).
        # - transformed_results_for_new_key: set by CopyStep (defaultdict(dict)). None only if CopyStep was
        #   never run (e.g. step run in isolation). In the full pipeline both are set.
        if context.postprocess_fn is None or context.transformed_results_for_new_key is None:
            return context
        for split, (transformed_chunks, metadata) in context._raw_transformed_by_split.items():
            postprocessed = context.postprocess_fn(transformed_chunks, metadata)
            for assign_to_key, assign_to_map in zip(
                context.assign_to_keys, context.assign_to_keys_map
            ):
                for chunk in postprocessed:
                    if assign_to_map not in chunk:
                        raise KeyError(
                            f"Transformed chunk does not contain the expected key '{assign_to_map}'. "
                            f"Available keys: {list(chunk.keys())}"
                        )
                context.transformed_results_for_new_key[assign_to_key][split] = [
                    chunk[assign_to_map] for chunk in postprocessed
                ]
        return context


class MergeStep:
    """
    Step 5: Merge per-split transformed results into context.transformed_data.

    What it does:
    - Calls context.strategy._merge_transformed_data(context.transformed_data,
      context.transformed_results_for_new_key, context.available_splits). The strategy writes
      each assign_to_key's split results into transformed_data[key][split], in place. After this,
      transformed_data holds the final transformed container returned to the caller.

    Depends on: CopyStep (transformed_data, available_splits, transformed_results_for_new_key
    structure), PostprocessStep (transformed_results_for_new_key filled), context.strategy.
    Affects: RecordManifestStep (reads transformed_data to append manifest entry); caller receives
    transformed_data as the pipeline result.
    """

    def run(self, context: TransformContext) -> TransformContext:
        strategy = context.strategy
        if (
            strategy is None
            or context.transformed_data is None
            or context.available_splits is None
            or context.transformed_results_for_new_key is None
        ):
            return context
        strategy._merge_transformed_data(
            base_data=context.transformed_data,
            transformed_results=context.transformed_results_for_new_key,
            available_splits=context.available_splits,
        )
        return context


def _get_producer_version() -> str:
    """Producer version for manifest entries (e.g. package version)."""
    try:
        import importlib.metadata
        return importlib.metadata.version("lmetk")
    except Exception:
        return "0.0.0"


class RecordManifestStep:
    """
    Step 6 (Phase 4.1): Append a transform metadata entry to the container's manifest, if present.

    What it does:
    - If context.transformed_data has a manifest attribute and it is a MetadataManifest, appends
      a ManifestEntry with category "transform" and payload (apply_to_keys, assign_to_keys,
      fit_on_split, fit_on_key, log_splits, optional slice_info). Does not change transformed_data
      contents; only the manifest. If there is no manifest, returns without change.

    Depends on: CopyStep (transformed_data exists and may carry a manifest from context.data copy).
    Affects: None for pipeline logic; enriches manifest for downstream consumers (traceability, debugging).
    """

    def run(self, context: TransformContext) -> TransformContext:
        if context.transformed_data is None:
            return context
        # manifest: optional MetadataManifest on the container (Phase 4.1). Set when the container is
        # created (e.g. datasource passes manifest=) or carried over by CopyStep from context.data.
        # If present, we append a transform entry (apply_to_keys, fit_on_split, etc.) for traceability.
        manifest = getattr(context.transformed_data, "manifest", None)
        if manifest is None or not isinstance(manifest, MetadataManifest):
            return context
        step_id = (
            context.step_id
            or getattr(
                context.transform_instance,
                "transform_name",
                context.transform_instance.__class__.__name__ if context.transform_instance else None,
            )
        )
        payload = {
            "transform_name": step_id,
            "transform_class": (
                context.transform_instance.__class__.__name__
                if context.transform_instance else None
            ),
            "apply_to_keys": context.apply_to_keys,
            "assign_to_keys": context.assign_to_keys,
            "fit_on_split": context.fit_on_split,
            "fit_on_key": context.fit_on_key,
            "log_splits": list(context.log.keys()) if context.log else [],
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
            payload["duration_seconds"] = round(context.finished_at - context.started_at, 6)
        if context.slice_info is not None:
            payload["slice_info"] = {
                "split": context.slice_info.split,
                "n_units": len(context.slice_info.unit_ids) if context.slice_info.unit_ids else None,
                "n_cycles": len(context.slice_info.cycle_ids) if context.slice_info.cycle_ids else None,
            }
        # entry.split: comma-joined list of splits this transform ran over. available_splits is set
        # in CopyStep from the data container (data[apply_to_keys[0]].keys()), i.e. the split names
        # (e.g. "train", "val", "test") come from whoever built the container (usually the datasource).
        split_str = (
            ",".join(context.available_splits)
            if getattr(context, "available_splits", None) else None
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
