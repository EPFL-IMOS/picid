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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from picid.data.data_objects import (
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
    """
    Return whether an object behaves like a sequence for this module.

    Parameters
    ----------
    obj : Any
        Object to inspect.

    Returns
    -------
    bool
        ``True`` for list/tuple/array-like inputs, but ``False`` for
        ``str`` and ``bytes``.
    """
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def postprocess_transformed_data(
    data: List[Union[np.ndarray, NamedTransformInput]] | ReturnObject,
    metadata: Dict[str, Any],
) -> List[Union[SimpleReturnObject, NamedDictReturnObject]]:
    """
    Standardize raw transform output into return objects with mapping keys.

    Parameters
    ----------
    data : list[numpy.ndarray | NamedTransformInput] | ReturnObject
        The raw output produced by a transform for a single split.
    metadata : dict[str, Any]
        Transform metadata containing ``assign_to_map``.

    Returns
    -------
    list[SimpleReturnObject | NamedDictReturnObject]
        Normalized outputs that expose the expected ``assign_to_map`` keys.

    Raises
    ------
    ValueError
        If the output shape cannot be normalized into a supported form.

    Notes
    -----
    ``PostprocessStep`` uses this helper to normalize the three output shapes a
    transform may return:

    1. Single array (one unit, single assign key).
    2. List of raw arrays (one per unit, single assign key).
    3. List of dict-like objects (one per unit, multiple assign keys).
    """
    assign_to_map = metadata.get("assign_to_map")
    assert is_sequence(
        assign_to_map
    ), f"Expected assign_to_map to be a list, got {type(assign_to_map)}"

    # Case 1: single non-mapping output
    if len(data) == 1 and not isinstance(data[0], Mapping):
        assert (
            len(assign_to_map) == 1
        ), "Expected single assign_to_map key for single output."
        return [SimpleReturnObject(**{assign_to_map[0]: data[0]})]

    # Case 2: multiple raw arrays
    if isinstance(data, Sequence) and all(
        not isinstance(chunk, Mapping) for chunk in data
    ):
        assert (
            len(assign_to_map) == 1
        ), "Expected single assign_to_map key for multiple outputs."
        return [SimpleReturnObject(**{assign_to_map[0]: chunk}) for chunk in data]

    # Case 3: list of dict-like objects
    if isinstance(data, Sequence) and all(isinstance(chunk, Mapping) for chunk in data):
        return [NamedDictReturnObject(**chunk) for chunk in data]

    raise ValueError(
        "Transformed data format not recognised. Expected a single array, "
        "a list of arrays, or a list of dict-like objects."
    )


class TransformStrategy:
    """
    Orchestrate the transform pipeline and expose chunk/merge helpers.

    The strategy is the single entry point for applying a transform to a
    ``SplitDatasetContainer``. It builds a ``TransformContext``, injects
    default integrity checks when ``validate_output=True``, runs the pipeline
    steps, and returns ``(transformed_data, log)``.
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
        """
        Build and raise a ``TransformError`` from the pipeline context.

        Parameters
        ----------
        context : TransformContext
            Pipeline context that captured the failure.
        e : Exception
            Original exception raised by the pipeline.
        """
        err = build_transform_error(context, e)
        raise err from e

    def _prepare_chunks_for_split(
        self,
        data_to_prepare: Dict[str, Any],
        split: str,
        unit_metadata_by_split: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[NamedTransformInput]:
        """
        Build one ``NamedTransformInput`` per unit for a split.

        Parameters
        ----------
        data_to_prepare : dict[str, Any]
            Split-keyed transform payload that still holds lists of units.
        split : str
            Split name to materialize into chunk wrappers.
        unit_metadata_by_split : dict[str, list[dict[str, Any]]], optional
            Optional per-split metadata aligned with the unit lists.

        Returns
        -------
        list[NamedTransformInput]
            Lightweight wrappers that keep references to the underlying arrays.

        Notes
        -----
        No sanitization is performed here; that is
        ``IntegrityCheckBeforeStep``'s responsibility. The caller is
        responsible for letting the returned list go out of scope when done so
        the wrappers can be garbage collected.

        This helper is called by ``RegisterChunkBuilderStep`` closures
        (primary path) and directly by ``FitStep`` / ``TransformStep`` as a
        fallback for test contexts where ``RegisterChunkBuilderStep`` did not
        run.
        """
        keys, num_units, unit_metadata_for_split = self._validate_chunk_alignment(
            data_to_prepare=data_to_prepare,
            split=split,
            unit_metadata_by_split=unit_metadata_by_split,
        )

        return [
            NamedTransformInput(
                # The chunk keeps the payload arrays by reference, but its
                # metadata is the per-unit metadata record for this exact unit.
                metadata=unit_metadata_for_split[i]
                if unit_metadata_for_split
                else None,
                **{key: data_to_prepare[key][split][i] for key in keys},
            )
            for i in range(num_units)
        ]

    @staticmethod
    def _validate_chunk_alignment(
        data_to_prepare: Dict[str, Any],
        split: str,
        unit_metadata_by_split: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[str], int, List[Dict[str, Any]]]:
        """
        Validate split-local payload counts before chunk materialization.

        Parameters
        ----------
        data_to_prepare : dict[str, Any]
            Split-keyed payload with unit lists for each transform input.
        split : str
            Split name that is being materialized.
        unit_metadata_by_split : dict[str, list[dict[str, Any]]], optional
            Optional metadata aligned with each unit list.

        Returns
        -------
        tuple[list[str], int, list[dict[str, Any]]]
            The keys, number of units, and aligned metadata for the split.
        """
        keys = list(data_to_prepare.keys())
        if not keys:
            return [], 0, []

        counts: dict[str, int] = {}
        for key in keys:
            if split not in data_to_prepare[key]:
                raise ValueError(
                    f"Split '{split}' is missing for payload key '{key}' during "
                    "chunk preparation."
                )
            split_value = data_to_prepare[key][split]
            if not isinstance(split_value, list):
                raise ValueError(
                    f"data['{key}']['{split}'] must be a list of units, got "
                    f"{type(split_value).__name__}."
                )
            counts[key] = len(split_value)

        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            raise ValueError(
                f"Misaligned unit counts for split '{split}' during chunk "
                f"preparation: {counts}."
            )

        num_units = unique_counts.pop() if unique_counts else 0
        unit_metadata_for_split = []
        if unit_metadata_by_split is not None:
            unit_metadata_for_split = unit_metadata_by_split.get(split, [])
            # We fail before chunk creation so schema/layout bugs surface as a
            # clear contract error instead of a later IndexError during
            # transform execution.
            if len(unit_metadata_for_split) != num_units:
                raise ValueError(
                    f"unit_metadata is misaligned for split '{split}': expected "
                    f"{num_units} entries but got {len(unit_metadata_for_split)}."
                )

        return keys, num_units, unit_metadata_for_split

    def _merge_transformed_data(
        self,
        base_data: SplitDatasetContainer,
        transformed_results: Dict[str, Dict[str, List[Any]]],
        available_splits: List[str],
    ) -> None:
        """
        Write transformed results into the container in place.

        Parameters
        ----------
        base_data : SplitDatasetContainer
            Container that receives the transformed results.
        transformed_results : dict[str, dict[str, list[Any]]]
            Nested mapping of assign-to keys and their split-local outputs.
        available_splits : list[str]
            Splits used to fill missing entries for newly created keys.

        Notes
        -----
        Existing keys are overwritten per split. New keys are created with
        entries for all available splits. Missing splits for new keys are
        stored as ``None`` so the container structure remains consistent.
        """
        for assign_to_key, split_results in transformed_results.items():
            if assign_to_key in base_data:
                for split, result in split_results.items():
                    base_data[assign_to_key][split] = result
            else:
                base_data[assign_to_key] = {
                    split: split_results.get(split) for split in available_splits
                }
