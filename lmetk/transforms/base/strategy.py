from typing import Any, Dict, List, Optional
import copy
from lmetk.data.data_objects.data import (
    DatasetContainer,
    OrderedInput,
    SimpleReturnObject,
    NamedDictReturnObject,
)

from collections import defaultdict
from collections.abc import Mapping, Sequence


def is_sequence(obj):
    """Return True for list/tuple/array-like, but False for str/bytes."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


class TransformStrategy:
    """
    Orchestrates how a transformation is applied to a DatasetContainer,
    with efficient copying and support for distinct single/multi-unit logic.
    """

    def apply(
        self,
        transform_instance: Any,
        data: "DatasetContainer",
        apply_to_keys: str | List[str],
        assign_to_keys: str | List[str],
        assign_to_keys_map: str | List[str] = None,
        fit_on_split: Optional[str] = None,
        fit_on_key: Optional[str] = None,
    ) -> "DatasetContainer":

        if isinstance(apply_to_keys, str):
            apply_to_keys = [apply_to_keys]

        if isinstance(assign_to_keys, str):
            assign_to_keys = [assign_to_keys]

        if isinstance(assign_to_keys_map, str):
            assign_to_keys_map = [assign_to_keys_map]

        # --- 1. EFFICIENT HYBRID COPY ---
        transformed_data = copy.copy(data)
        keys_to_modify = apply_to_keys + (assign_to_keys if assign_to_keys else [])
        for key in keys_to_modify:
            if key in data:
                transformed_data[key] = copy.deepcopy(data[key])

        is_multi_unit = data.is_multi_unit
        available_splits = list(data[apply_to_keys[0]].keys())

        # --- 2. FIT TRANSFORMER (with SINGLE/MULTI LOGIC) ---

        if fit_on_split and fit_on_key:
            fit_chunks = self._prepare_chunks_for_split(
                {fit_on_key: data[fit_on_key]}, fit_on_split, is_multi_unit
            )
            metadata = {
                "mode": fit_on_split,
                "apply_to_keys": fit_on_key,
                "assign_to_keys": assign_to_keys,
                "assign_to_map": assign_to_keys_map,
            }

            if is_multi_unit:
                transform_instance.fit_multi_source(fit_chunks, metadata=metadata)
            else:
                transform_instance.fit_data(fit_chunks[0], metadata=metadata)

        # --- 3. TRANSFORM ALL SPLITS ---
        target_data_to_transform = {key: data[key] for key in apply_to_keys}

        # This block handles both merging strategies
        transformed_results_for_new_key = defaultdict(dict)
        for split in available_splits:
            chunks_to_transform = self._prepare_chunks_for_split(
                target_data_to_transform, split, is_multi_unit
            )

            metadata = {
                "mode": split,
                "apply_to_keys": apply_to_keys,
                "assign_to_keys": assign_to_keys,
                "assign_to_map": assign_to_keys_map,
            }
            # Call the appropriate transform method
            if is_multi_unit:
                transformed_chunks = transform_instance.transform_multi_source(
                    chunks_to_transform, metadata=metadata
                )
            else:
                transformed_chunks = [
                    transform_instance.transform_data(
                        chunks_to_transform[0], metadata=metadata
                    )
                ]

            transformed_chunks = self._postprocess_transformed_data(
                transformed_chunks, metadata
            )

            # --- 4. MERGE RESULTS (IN-PLACE OR NEW KEY) --- #

            # Mode 1: Assign results to a key defined by assign_to_keys
            for assign_to_key, assign_to_map in zip(assign_to_keys, assign_to_keys_map):
                transformed_results_for_new_key[assign_to_key][split] = [
                    chunk[assign_to_map] for chunk in transformed_chunks
                ]
                # if is_multi_unit:
                # else:
                #     assert len(transformed_chunks) == 1, "Expected single chunk for non-multi-unit data."
                #     transformed_results_for_new_key[assign_to_key][split] = transformed_chunks[assign_to_map]

        for assign_to_key, splits in transformed_results_for_new_key.items():
            for split, data in splits.items():
                # Check whenever assign_to_key exist in transformed_data.keys()
                if assign_to_key not in transformed_data:
                    transformed_data[assign_to_key] = {
                        "train": None,
                        "val": None,
                        "test": None,
                    }

                transformed_data[assign_to_key][split] = (
                    transformed_results_for_new_key[assign_to_key][split]
                )

        return transformed_data

    def _postprocess_transformed_data(self, data, metadata):
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
        if isinstance(data, Sequence) and all(
            isinstance(chunk, Mapping) for chunk in data
        ):
            return [NamedDictReturnObject(**chunk) for chunk in data]

        raise ValueError("Transformed data format not recognized or not handled.")

    def _prepare_chunks_for_split(
        self, data_to_prepare: Dict, split: str, is_multi_unit: bool
    ) -> List["OrderedInput"]:

        keys = list(data_to_prepare.keys())

        # For multi-unit data, extract each unit for each key
        chunks = []
        num_units = len(data_to_prepare[keys[0]][split])

        if is_multi_unit == True:
            assert (
                num_units > 1
            ), "Flag 'is_multi_unit' is true hence expect multiple units."

        # TODO: why is this ordered input, when we go by key?
        for i in range(num_units):
            unit_data = {
                key: data_to_prepare[key][split][i]
                for key in keys
                if split in data_to_prepare[key]
            }
            chunks.append(OrderedInput(**unit_data).get_sanitized_data())
        return chunks
