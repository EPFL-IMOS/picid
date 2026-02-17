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
    """Return True for list/tuple/array-like, but False for str/bytes."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def postprocess_transformed_data(
    data: List[Union[np.ndarray, NamedTransformInput]] | ReturnObject,
    metadata: dict[str, Any],
):
    """
    Standardizes the output of the transform step into a list of SimpleReturnObject.
    Handles cases where the transform returns a single array or a list of arrays.
    """
    assign_to_map = metadata.get("assign_to_map")
    assert is_sequence(
        assign_to_map
    ), f"Expected assign_to_map to be a list. Got {type(assign_to_map)}"

    # ---Simple Return Objects---
    # Case 1: Single output (e.g., np.ndarray)
    if len(data) == 1 and not isinstance(data[0], Mapping):
        assert (
            len(assign_to_map) == 1
        ), "Expected single assign_to_map for single output."
        return [SimpleReturnObject(**{assign_to_map[0]: data[0]})]

    # Case 2: Multiple outputs (e.g., list of np.ndarray)
    if isinstance(data, Sequence) and all(
        not isinstance(chunk, Mapping) for chunk in data
    ):
        assert (
            len(assign_to_map) == 1
        ), "Expected single assign_to_map for multiple outputs."
        return [SimpleReturnObject(**{assign_to_map[0]: chunk}) for chunk in data]

    # ---Named Return Objects---
    # Case 3: Already in expected format (list of Mapping/SimpleReturnObject)
    if isinstance(data, Sequence) and all(isinstance(chunk, Mapping) for chunk in data):
        return [NamedDictReturnObject(**chunk) for chunk in data]

    raise ValueError("Transformed data format not recognized or not handled.")


class TransformStrategy:
    """
    Orchestrates how a transformation is applied to a DatasetContainer,
    with efficient copying and support for distinct single/multi-unit logic.
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
        if isinstance(apply_to_keys, str):
            apply_to_keys = [apply_to_keys]
        if isinstance(assign_to_keys, str):
            assign_to_keys = [assign_to_keys]
        if isinstance(assign_to_keys_map, str):
            assign_to_keys_map = [assign_to_keys_map]

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
        except Exception:
            _emit("on_transform_error", context)
            raise
        _emit("on_pipeline_end", context)
        return context.transformed_data, context.log

    def _merge_transformed_data(self, base_data, transformed_results, available_splits):

        for assign_to_key in transformed_results.keys():
            if assign_to_key in base_data:
                for split in transformed_results[assign_to_key].keys():
                    base_data[assign_to_key][split] = transformed_results[
                        assign_to_key
                    ][split]
            else:
                base_data[assign_to_key] = {
                    split: transformed_results[assign_to_key][split]
                    for split in available_splits
                    # "train": transformed_results[assign_to_key].get("train"),
                    # "val": transformed_results[assign_to_key].get("val"),
                    # "test": transformed_results[assign_to_key].get("test"),
                }

    def _prepare_chunks_for_split(
        self, data_to_prepare: Dict, split: str, is_multi_unit: bool
    ) -> List["NamedTransformInput"]:

        keys = list(data_to_prepare.keys())

        # For multi-unit data, extract each unit for each key
        chunks = []
        num_units = len(data_to_prepare[keys[0]][split])

        if is_multi_unit == True:
            assert isinstance(
                data_to_prepare[keys[0]][split], list
            ), "Flag 'is_multi_unit' is true, so expected a list for each split."

        # TODO: why is this ordered input, when we go by key?
        for i in range(num_units):
            unit_data = {
                key: data_to_prepare[key][split][i]
                for key in keys
                if split in data_to_prepare[key]
            }

            chunks.append(NamedTransformInput(**unit_data).get_sanitized_data())
        return chunks
