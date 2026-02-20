"""
Pipeline context and type contracts.

This module defines the shared state (TransformContext) passed through every
pipeline step, and the type aliases used by the integrity-check system.

Relationship to the rest of the pipeline:
- TransformStrategy (in strategy/) builds a TransformContext and runs the steps
  defined in pipeline/steps.py in order.
- Each step receives the context, mutates it (or reads from it), and returns it.
- The hook system (hooks_registry) and default checks (checks) use TransformContext
  for observability and validation; steps use it as the single source of truth
  for data, assign_to keys, and chunk builders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from picid.data.data_objects.data import NamedTransformInput, SplitDatasetContainer
from picid.data.data_objects.slice_info import SliceInfo

# ---------------------------------------------------------------------------
# Integrity check type aliases
# ---------------------------------------------------------------------------
# These describe the contract for before/after checks injected by the strategy
# and run by IntegrityCheckBeforeStep / IntegrityCheckAfterStep (see checks.py).

# Before-checks receive one split's chunks: Dict[split_name, List[NamedTransformInput]].
BeforeCheckFn = Callable[[Dict[str, List[NamedTransformInput]]], None]
# After-checks receive only the assign_to slice: Dict[key, value] for keys this transform wrote.
AfterCheckFn = Callable[[Dict[str, Any]], None]
# A check is (name, function, policy) with policy in {"fail", "warn", "allow"}.
BeforeCheck = Tuple[str, BeforeCheckFn, str]
AfterCheck = Tuple[str, AfterCheckFn, str]


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
    build_fit_chunks, build_transform_chunks.

    Mutable pipeline state (set by steps in order)
    -----------------------------------------------
    transformed_data, available_splits, transformed_results_for_new_key,
    _raw_transformed_by_split, log.

    Integrity run state (set by integrity steps; read by RecordManifestStep)
    -------------------------------------------------------------------------
    integrity_before_ran, integrity_before_check_names,
    integrity_after_ran, integrity_after_check_names.
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
    integrity_checks: Optional[List[Tuple[str, Callable[[], None], str]]] = None

    # --- Chunk builders (set by RegisterChunkBuilderStep) ---
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


class PipelineStep(Protocol):
    """
    Protocol for pipeline steps.

    All concrete steps (CopyStep, FitStep, TransformStep, etc.) in steps.py
    implement this interface: they receive the shared context, update it
    in place or read from it, and return it so the next step can run.
    The strategy calls step.run(context) for each step in sequence.
    """

    def run(self, context: TransformContext) -> TransformContext:
        """Execute this step; may mutate context. Returns context for chaining."""
        ...
