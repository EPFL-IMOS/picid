"""
Transform strategy: orchestrates the transform pipeline and provides chunk preparation / merge helpers.

TransformStrategy.apply() builds a TransformContext, runs the pipeline steps (Copy → IntegrityCheck
→ Fit → Transform → Postprocess → Merge → RecordManifest), and returns (transformed_data, log).
The strategy is also attached to the context so steps can call _prepare_chunks_for_split (to get
per-unit chunks for a split) and _merge_transformed_data (to write transformed results back into
the container). postprocess_transformed_data standardizes raw transform output into a list of
SimpleReturnObject / NamedDictReturnObject for MergeStep.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from lmetk.data.data_objects.data import (
    NamedTransformInput,
    ReturnObject,
    SimpleReturnObject,
    NamedDictReturnObject,
    SplitDatasetContainer,
)

from collections import defaultdict
from collections.abc import Mapping, Sequence
from lmetk.utils.decorators import inject_transform_context_to_strategy_apply
from lmetk.transforms.base.pipeline import (
    TransformContext,
    CopyStep,
    IntegrityCheckStep,
    FitStep,
    TransformStep,
    PostprocessStep,
    MergeStep,
    RecordManifestStep,
    _emit,
)


def is_sequence(obj):
    """
    Return True for list/tuple/array-like, but False for str/bytes.

    Used in postprocess_transformed_data to check assign_to_map and to distinguish
    list-of-arrays from a single array. We exclude str/bytes so that string metadata
    is not treated as a sequence of items.
    """
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def postprocess_transformed_data(
    data: List[Union[np.ndarray, NamedTransformInput]] | ReturnObject,
    metadata: dict[str, Any],
):
    """
    Standardize raw transform output into a list of objects with assign_to_map keys.

    Called by PostprocessStep for each split. The transform may return a single array,
    a list of arrays (one per unit), or already a list of dict-like objects. We
    normalize to a list of SimpleReturnObject or NamedDictReturnObject so that
    PostprocessStep can read chunk[assign_to_map] for each chunk and MergeStep
    receives a consistent structure.

    Parameters
    ----------
    data : list of arrays or list of Mapping
        Raw output from transform_data / transform_multi_source for one split.
    metadata : dict
        Must contain "assign_to_map" (list of keys to assign, e.g. ["features"]).

    Returns
    -------
    list of SimpleReturnObject or NamedDictReturnObject
        One item per unit/chunk, each with the key(s) from assign_to_map.

    Raises
    ------
    ValueError
        If data format is not one of the supported cases.
    """
    assign_to_map = metadata.get("assign_to_map")
    assert is_sequence(
        assign_to_map
    ), f"Expected assign_to_map to be a list. Got {type(assign_to_map)}"

    # --- Simple return: transform returned a single array (e.g. one unit) ---
    if len(data) == 1 and not isinstance(data[0], Mapping):
        assert (
            len(assign_to_map) == 1
        ), "Expected single assign_to_map for single output."
        return [SimpleReturnObject(**{assign_to_map[0]: data[0]})]

    # --- Multiple raw arrays: one chunk per unit, same assign_to_map key for all ---
    if isinstance(data, Sequence) and all(
        not isinstance(chunk, Mapping) for chunk in data
    ):
        assert (
            len(assign_to_map) == 1
        ), "Expected single assign_to_map for multiple outputs."
        return [SimpleReturnObject(**{assign_to_map[0]: chunk}) for chunk in data]

    # --- Transform already returned list of dict-like objects (e.g. NamedDictReturnObject) ---
    if isinstance(data, Sequence) and all(isinstance(chunk, Mapping) for chunk in data):
        return [NamedDictReturnObject(**chunk) for chunk in data]

    raise ValueError("Transformed data format not recognized or not handled.")


class TransformStrategy:
    """
    Orchestrates the transform pipeline and provides chunk/merge helpers for the steps.

    The strategy is the single entry point for applying a transform to a SplitDatasetContainer.
    It builds a TransformContext (with strategy=self and postprocess_fn=postprocess_transformed_data),
    runs the pipeline steps in order, and returns (transformed_data, log). Pipeline steps use
    context.strategy to call:
    - _prepare_chunks_for_split: turn data[key][split] into a list of NamedTransformInput (one per
      unit), so FitStep and TransformStep can pass one chunk at a time to the transform.
    - _merge_transformed_data: write transformed_results (assign_to_key -> split -> list of arrays)
      into the container in place, so MergeStep can complete the pipeline output.
    """

    @inject_transform_context_to_strategy_apply
    def apply(
        self,
        transform_instance: Any,
        data: SplitDatasetContainer,
        apply_to_keys: str | List[str],
        assign_to_keys: str | List[str],
        assign_to_keys_map: str | List[str] = None,
        fit_on_split: Optional[str] = None,
        fit_on_key: Optional[str] = None,
        transform_on_keys: Optional[list[str]] = None,
    ):
        # Normalize single string to list so context and steps always see lists
        if isinstance(apply_to_keys, str):
            apply_to_keys = [apply_to_keys]
        if isinstance(assign_to_keys, str):
            assign_to_keys = [assign_to_keys]
        if isinstance(assign_to_keys_map, str):
            assign_to_keys_map = [assign_to_keys_map]

        # Build context: pipeline steps will read and mutate this. strategy=self so steps can
        # call _prepare_chunks_for_split and _merge_transformed_data.
        context = TransformContext(
            data=data,
            transform_instance=transform_instance,
            apply_to_keys=apply_to_keys,
            assign_to_keys=assign_to_keys,
            assign_to_keys_map=assign_to_keys_map,
            fit_on_split=fit_on_split,
            fit_on_key=fit_on_key,
            transform_on_keys=transform_on_keys,
            strategy=self,
            postprocess_fn=postprocess_transformed_data,
            slice_info=getattr(data, "slice_info", None),
        )
        _emit("on_pipeline_start", context)
        try:
            CopyStep().run(context)
            IntegrityCheckStep().run(context)
            FitStep().run(context)
            TransformStep().run(context)
            PostprocessStep().run(context)
            MergeStep().run(context)
            RecordManifestStep().run(context)
        except Exception as e:
            context.error = e
            _emit("on_transform_error", context)
            raise
        _emit("on_pipeline_end", context)
        return context.transformed_data, context.log

    def _merge_transformed_data(self, base_data, transformed_results, available_splits):
        """
        Write transformed results into the container in place. Called by MergeStep.

        For each assign_to_key in transformed_results we either overwrite existing
        base_data[key][split] or create a new key and fill it for each split in
        available_splits (so the container has a consistent set of splits per key).
        """
        for assign_to_key in transformed_results.keys():
            if assign_to_key in base_data:
                # Key already exists (e.g. in-place transform on "features"): overwrite per split
                for split in transformed_results[assign_to_key].keys():
                    base_data[assign_to_key][split] = transformed_results[
                        assign_to_key
                    ][split]
            else:
                # New key (e.g. assign_to "rul" when data had "target"): create key with all splits
                base_data[assign_to_key] = {
                    split: transformed_results[assign_to_key][split]
                    for split in available_splits
                }

    def _prepare_chunks_for_split(
        self, data_to_prepare: Dict, split: str, is_multi_unit: bool
    ) -> List["NamedTransformInput"]:
        """
        Build one NamedTransformInput per unit for the given split. Called by FitStep and TransformStep.

        data_to_prepare is keyed by data key (e.g. "features", "target"); each value is
        split -> list of per-unit arrays. We iterate by unit index i and collect
        data_to_prepare[key][split][i] for each key into one NamedTransformInput, so the
        transform receives one chunk (one unit) at a time with all keys present.
        """
        keys = list(data_to_prepare.keys())
        chunks = []
        num_units = len(data_to_prepare[keys[0]][split])

        if is_multi_unit == True:
            assert isinstance(
                data_to_prepare[keys[0]][split], list
            ), "Flag 'is_multi_unit' is true, so expected a list for each split."

        # One chunk per unit: chunk i has data_to_prepare[key][split][i] for each key
        for i in range(num_units):
            unit_data = {
                key: data_to_prepare[key][split][i]
                for key in keys
                if split in data_to_prepare[key]
            }
            chunks.append(NamedTransformInput(**unit_data).get_sanitized_data())
        return chunks
