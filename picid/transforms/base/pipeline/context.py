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

from picid.data.data_objects import NamedTransformInput, SplitDatasetContainer
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

    Attributes
    ----------
    data : SplitDatasetContainer
        Input data container being transformed.
    transform_instance : Any
        Transform object executed by the pipeline.
    apply_to_keys : list[str]
        Keys consumed by the transform.
    assign_to_keys : list[str]
        Keys populated by the transform output.
    assign_to_keys_map : list[str]
        Mapping used by the postprocess step.
    fit_on_split : str, optional
        Split used for fitting when applicable.
    fit_on_key : str, optional
        Key used for fitting when applicable.
    transform_on_keys : list[str], optional
        Keys processed during the transform stage.
    include_slice_info_in_metadata : bool
        Whether slice information should be injected into metadata.
    strategy : Any, optional
        Strategy object orchestrating the pipeline.
    postprocess_fn : callable, optional
        Function that normalizes raw transform outputs.
    slice_info : SliceInfo, optional
        Slice metadata passed through the pipeline when available.
    step_id : str, optional
        Optional transform identifier used in errors and logs.
    integrity_checks_before : list[BeforeCheck], optional
        Before-checks scheduled by the strategy.
    integrity_checks_after : list[AfterCheck], optional
        After-checks scheduled by the strategy.
    integrity_checks : list[tuple[str, Callable[[], None], str]], optional
        Legacy combined integrity-check list.
    build_fit_chunks : callable, optional
        Closure that builds fit-time chunks.
    build_transform_chunks : callable, optional
        Closure that builds transform-time chunks.
    log : dict[str, Any]
        Mutable pipeline log collected during execution.
    transformed_data : SplitDatasetContainer, optional
        Output container being assembled by the pipeline.
    available_splits : list[str], optional
        Splits currently available for the transform output.
    transformed_results_for_new_key : dict[str, dict[str, list[Any]]], optional
        Intermediate results for newly created output keys.
    _raw_transformed_by_split : dict[str, Any]
        Internal split-local raw outputs before merge.
    integrity_before_ran : bool
        Whether before-checks ran.
    integrity_before_check_names : list[str]
        Names of before-checks that ran.
    integrity_after_ran : bool
        Whether after-checks ran.
    integrity_after_check_names : list[str]
        Names of after-checks that ran.
    started_at : float, optional
        Pipeline start timestamp.
    finished_at : float, optional
        Pipeline end timestamp.
    cache_status : str, optional
        Cache-related status if the pipeline records one.
    error : BaseException, optional
        Captured pipeline error, if any.
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
    include_slice_info_in_metadata: bool = False

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
        """
        Execute this step and return the updated context.

        Parameters
        ----------
        context : TransformContext
            Shared pipeline context.

        Returns
        -------
        TransformContext
            The updated pipeline context.
        """
        ...
