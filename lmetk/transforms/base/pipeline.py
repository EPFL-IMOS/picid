"""
Phase 3: Composable pipeline steps and lifecycle hooks.

- TransformContext: holds data, transform_instance, metadata, log; passed through the pipeline.
- PipelineStep: protocol run(context) -> context.
- Concrete steps: CopyStep, FitStep, TransformStep, PostprocessStep, MergeStep.
- Hook registry and events: on_pipeline_start, before_transform, after_transform, on_transform_error, on_pipeline_end.
"""
from __future__ import annotations

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
    Context passed through the transform pipeline. Steps read and update this object.

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
    # Mutable state updated by steps
    transformed_data : Optional[SplitDatasetContainer]
        Set by CopyStep; holds the copy that will receive merged results.
    available_splits : Optional[List[str]]
        Set by CopyStep; list of split names to transform.
    transformed_results_for_new_key : Optional[Dict]
        Set by TransformStep; assign_to_key -> split -> list of arrays.
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

    # Set by strategy when building context so steps can call _prepare_chunks_for_split, _merge_transformed_data
    strategy: Optional[Any] = None
    # Postprocess function (strategy.postprocess_transformed_data) set by strategy
    postprocess_fn: Optional[Callable[..., Any]] = None
    # Slice-awareness (Phase 4.2): copied from data.slice_info when building context; read-only for transforms
    slice_info: Optional[SliceInfo] = None
    # Integrity (Phase 4.3): optional list of (name, check_fn, policy); IntegrityCheckStep runs them
    integrity_checks: Optional[List[Tuple[str, Callable[[], None], str]]] = None


# -----------------------------------------------------------------------------
# PipelineStep protocol
# -----------------------------------------------------------------------------


class PipelineStep(Protocol):
    """Protocol for a pipeline step: run(context) -> context."""

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


# -----------------------------------------------------------------------------
# Concrete pipeline steps (Phase 3.2)
# -----------------------------------------------------------------------------

from collections import defaultdict

import awkward as ak


class CopyStep:
    """Step 1: Hybrid copy and validation. Sets transformed_data, available_splits."""

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

        context.transformed_data = transformed_data
        context.available_splits = available_splits
        context.transformed_results_for_new_key = defaultdict(dict)
        context._raw_transformed_by_split = {}
        return context


class IntegrityCheckStep:
    """Step 1.5 (Phase 4.3): Run optional integrity checks (policy: fail / warn / allow). No-op if context.integrity_checks is None."""

    def run(self, context: TransformContext) -> TransformContext:
        if not context.integrity_checks:
            return context
        run_checks(context.integrity_checks)
        return context


class FitStep:
    """Step 2: Fit transform if fit_on_split and fit_on_key are set."""

    def run(self, context: TransformContext) -> TransformContext:
        if not context.fit_on_split or not context.fit_on_key:
            return context
        strategy = context.strategy
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
    """Step 3: Transform each split; store raw results in _raw_transformed_by_split."""

    def run(self, context: TransformContext) -> TransformContext:
        strategy = context.strategy
        if strategy is None or context.transformed_data is None or context.available_splits is None:
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
    """Step 4: Postprocess raw transform output per split and fill transformed_results_for_new_key."""

    def run(self, context: TransformContext) -> TransformContext:
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
    """Step 5: Merge transformed results into transformed_data."""

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
    """Step 6 (Phase 4.1): If container has a manifest, append a transform metadata entry."""

    def run(self, context: TransformContext) -> TransformContext:
        if context.transformed_data is None:
            return context
        manifest = getattr(context.transformed_data, "manifest", None)
        if manifest is None or not isinstance(manifest, MetadataManifest):
            return context
        step_id = getattr(
            context.transform_instance,
            "transform_name",
            context.transform_instance.__class__.__name__ if context.transform_instance else None,
        )
        payload = {
            "apply_to_keys": context.apply_to_keys,
            "assign_to_keys": context.assign_to_keys,
            "fit_on_split": context.fit_on_split,
            "fit_on_key": context.fit_on_key,
            "log_splits": list(context.log.keys()) if context.log else [],
        }
        if context.slice_info is not None:
            payload["slice_info"] = {
                "split": context.slice_info.split,
                "n_units": len(context.slice_info.unit_ids) if context.slice_info.unit_ids else None,
                "n_cycles": len(context.slice_info.cycle_ids) if context.slice_info.cycle_ids else None,
            }
        entry = ManifestEntry(
            schema_version=MANIFEST_SCHEMA_VERSION,
            producer_version=_get_producer_version(),
            category="transform",
            payload=payload,
            step_id=step_id,
            key=context.apply_to_keys[0] if context.apply_to_keys else None,
        )
        manifest.add(entry)
        return context
