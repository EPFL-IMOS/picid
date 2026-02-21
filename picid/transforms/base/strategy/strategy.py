"""
strategy.py
===========

TransformStrategy orchestrates the transform pipeline and provides chunk
preparation / merge helpers used by pipeline steps.

Pipeline order:
    CopyStep
    → RegisterChunkBuilderStep
    → IntegrityCheckBeforeStep
    → FitStep
    → TransformStep
    → PostprocessStep
    → MergeStep
    → IntegrityCheckAfterStep
    → RecordManifestStep

Memory model
------------
_prepare_chunks_for_split builds NamedTransformInput wrappers that hold
*references* to the underlying arrays — no array data is copied. Chunks
are built on demand per split by the registered callables and discarded
after each step's local scope exits. Peak memory is one split's chunk
wrappers at any time.

Default integrity checks
------------------------
When validate_output=True (the default):
- Before-checks: NaN detection + length consistency on apply_to chunks.
- After-checks:  structural validity + split-set preservation on assign_to output.

These replace DataTransform._validate_forward_output entirely. Soft diagnostic
warnings (type change, unit-count change) live in MergeStep.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from picid.data.data_objects.data import (
    NamedTransformInput,
    NamedDictReturnObject,
    ReturnObject,
    SimpleReturnObject,
    SplitDatasetContainer,
)
from picid.exceptions import TransformError, build_transform_error
from picid.transforms.base.pipeline import (
    TransformContext,
    CopyStep,
    RegisterChunkBuilderStep,
    IntegrityCheckBeforeStep,
    IntegrityCheckAfterStep,
    FitStep,
    TransformStep,
    PostprocessStep,
    MergeStep,
    RecordManifestStep,
    build_default_before_checks,
    build_default_after_checks,
    _emit,
)
from picid.utils.decorators import inject_transform_context_to_strategy_apply


def is_sequence(obj):
    """Return True for list/tuple/array-like, but False for str/bytes."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def postprocess_transformed_data(
    data: List[Union[np.ndarray, NamedTransformInput]] | ReturnObject,
    metadata: Dict[str, Any],
) -> List[Union[SimpleReturnObject, NamedDictReturnObject]]:
    """
    Standardize raw transform output into a list of objects with assign_to_map keys.

    Called by PostprocessStep for each split. Normalises the three shapes a
    transform may return:
        1. Single array (one unit, single assign key).
        2. List of raw arrays (one per unit, single assign key).
        3. List of dict-like objects (one per unit, multiple assign keys).

    Raises ValueError if the format is not recognised.
    """
    assign_to_map = metadata.get("assign_to_map")
    assert is_sequence(assign_to_map), (
        f"Expected assign_to_map to be a list, got {type(assign_to_map)}"
    )

    # Case 1: single non-mapping output
    if len(data) == 1 and not isinstance(data[0], Mapping):
        assert len(assign_to_map) == 1, (
            "Expected single assign_to_map key for single output."
        )
        return [SimpleReturnObject(**{assign_to_map[0]: data[0]})]

    # Case 2: multiple raw arrays
    if isinstance(data, Sequence) and all(
        not isinstance(chunk, Mapping) for chunk in data
    ):
        assert len(assign_to_map) == 1, (
            "Expected single assign_to_map key for multiple outputs."
        )
        return [SimpleReturnObject(**{assign_to_map[0]: chunk}) for chunk in data]

    # Case 3: list of dict-like objects
    if isinstance(data, Sequence) and all(
        isinstance(chunk, Mapping) for chunk in data
    ):
        return [NamedDictReturnObject(**chunk) for chunk in data]

    raise ValueError(
        "Transformed data format not recognised. Expected a single array, "
        "a list of arrays, or a list of dict-like objects."
    )


class TransformStrategy:
    """
    Orchestrates the transform pipeline and provides chunk/merge helpers.

    The strategy is the single entry point for applying a transform to a
    SplitDatasetContainer. It builds a TransformContext, injects default
    integrity checks when validate_output=True, runs the pipeline steps,
    and returns (transformed_data, log).
    """

    @inject_transform_context_to_strategy_apply
    def apply(
        self,
        transform_instance: Any,
        data: SplitDatasetContainer,
        apply_to_keys: Union[str, List[str]],
        assign_to_keys: Union[str, List[str]],
        assign_to_keys_map: Union[str, List[str]] = None,
        fit_on_split: Optional[str] = None,
        fit_on_key: Optional[str] = None,
        transform_on_keys: Optional[List[str]] = None,
        step_id: Optional[str] = None,
        integrity_checks_before: Optional[List[Tuple]] = None,
        integrity_checks_after: Optional[List[Tuple]] = None,
        validate_output: bool = True,
        include_slice_info_in_metadata: bool = False,
    ) -> Tuple[SplitDatasetContainer, Dict[str, Any]]:

        # --- Normalise key lists ---
        if isinstance(apply_to_keys, str):
            apply_to_keys = [apply_to_keys]
        if isinstance(assign_to_keys, str):
            assign_to_keys = [assign_to_keys]
        if isinstance(assign_to_keys_map, str):
            assign_to_keys_map = [assign_to_keys_map]

        # --- Resolve default checks when validate_output=True ---
        # original_splits is captured here (before the transform runs) so the
        # after-check can compare against it.
        original_splits: frozenset = frozenset()
        if apply_to_keys and apply_to_keys[0] in data:
            first_apply = data[apply_to_keys[0]]
            if not isinstance(first_apply, Mapping):
                raise TypeError(
                    f"data['{apply_to_keys[0]}'] must be a split-keyed mapping, "
                    f"got {type(first_apply).__name__}."
                )
            original_splits = frozenset(first_apply.keys())

        if validate_output:
            if integrity_checks_before is None:
                integrity_checks_before = build_default_before_checks(policy="fail")
            if integrity_checks_after is None:
                integrity_checks_after = build_default_after_checks(
                    original_splits=original_splits, policy="fail"
                )

        # --- Build context ---
        context = TransformContext(
            data=data,
            transform_instance=transform_instance,
            apply_to_keys=apply_to_keys,
            assign_to_keys=assign_to_keys,
            assign_to_keys_map=assign_to_keys_map,
            fit_on_split=fit_on_split,
            fit_on_key=fit_on_key,
            transform_on_keys=transform_on_keys,
            include_slice_info_in_metadata=include_slice_info_in_metadata,
            strategy=self,
            postprocess_fn=postprocess_transformed_data,
            slice_info=getattr(data, "slice_info", None),
            step_id=step_id,
            integrity_checks_before=integrity_checks_before,
            integrity_checks_after=integrity_checks_after,
        )

        # --- Run pipeline ---
        _emit("on_pipeline_start", context)
        try:
            _emit("before_transform", context)
            CopyStep().run(context)
            RegisterChunkBuilderStep().run(context)
            IntegrityCheckBeforeStep().run(context)
            FitStep().run(context)
            TransformStep().run(context)
            PostprocessStep().run(context)
            MergeStep().run(context)
            IntegrityCheckAfterStep().run(context)
            _emit("after_transform", context)
            RecordManifestStep().run(context)
        except Exception as e:
            context.error = e
            _emit("on_transform_error", context)
            if isinstance(context.error, TransformError):
                raise context.error from context.error.cause
            self._raise_transform_error(context, e)

        _emit("on_pipeline_end", context)
        return context.transformed_data, context.log

    # ------------------------------------------------------------------
    # Helpers used by pipeline steps
    # ------------------------------------------------------------------

    def _raise_transform_error(self, context: TransformContext, e: Exception) -> None:
        """Build and raise TransformError from pipeline context and caught exception."""
        err = build_transform_error(context, e)
        raise err from e

    def _prepare_chunks_for_split(
        self,
        data_to_prepare: Dict[str, Any],
        split: str,
        is_multi_unit: bool,
    ) -> List[NamedTransformInput]:
        """
        Build one NamedTransformInput per unit for the given split.

        Returns lightweight wrappers holding *references* to the underlying
        arrays — no array data is copied. The caller is responsible for
        letting the returned list go out of scope when done so the wrappers
        can be GC'd.

        No sanitization is performed here; that is IntegrityCheckBeforeStep's
        responsibility.

        Called by RegisterChunkBuilderStep's closures (primary path) and
        directly by FitStep/TransformStep as a fallback for test contexts
        where RegisterChunkBuilderStep did not run.
        """
        keys = list(data_to_prepare.keys())
        if not keys:
            return []

        num_units = len(data_to_prepare[keys[0]][split])

        if is_multi_unit:
            assert isinstance(data_to_prepare[keys[0]][split], list), (
                "is_multi_unit=True but split data is not a list."
            )

        return [
            NamedTransformInput(**{
                key: data_to_prepare[key][split][i]
                for key in keys
                if split in data_to_prepare[key]
            })
            for i in range(num_units)
        ]

    def _merge_transformed_data(
        self,
        base_data: SplitDatasetContainer,
        transformed_results: Dict[str, Dict[str, List[Any]]],
        available_splits: List[str],
    ) -> None:
        """
        Write transformed results into the container in place.

        Existing keys: overwrite per split.
        New keys: create with entries for all available splits.
        Missing splits for new keys are stored as None (structure is consistent).
        """
        for assign_to_key, split_results in transformed_results.items():
            if assign_to_key in base_data:
                for split, result in split_results.items():
                    base_data[assign_to_key][split] = result
            else:
                base_data[assign_to_key] = {
                    split: split_results.get(split)
                    for split in available_splits
                }
