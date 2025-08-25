from importlib import metadata
import warnings
from typing import Any, Dict, List, Optional
import copy
from lmetk.data.data_objects.data import DatasetContainer, OrderedInput, SimpleReturnObject

from collections import defaultdict
from collections.abc import Mapping

class TransformStrategy():
    """
    Orchestrates how a transformation is applied to a DatasetContainer,
    with efficient copying and support for distinct single/multi-unit logic.
    """
    def apply(
        self,
        transform_instance: Any,
        data: 'DatasetContainer',
        apply_to_keys: str | List[str],
        assign_to_keys: str | List[str],
        assign_to_keys_map: str | List[str] = None,
        fit_on_split: Optional[str] = None,
        fit_on_key: Optional[str] = None,
    ) -> 'DatasetContainer':
        
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
            fit_on_data = {fit_on_key: data[fit_on_key], "metadata": data["metadata"]}
            fit_chunks = self._prepare_chunks_for_split(fit_on_data, fit_on_split, is_multi_unit)
            metadata={"mode": fit_on_split, "apply_to_keys": fit_on_key, "assign_to_keys":assign_to_keys, "assign_to_map": assign_to_keys_map}
            
            if is_multi_unit:
                transform_instance.fit_multi_source(fit_chunks, metadata=metadata)
            else:
                transform_instance.fit_data(fit_chunks[0], metadata=metadata)

        # --- 3. TRANSFORM ALL SPLITS ---
        target_data_to_transform = {key: data[key] for key in apply_to_keys}
        
        # This block handles both merging strategies
        transformed_results_for_new_key = defaultdict(dict)
        for split in available_splits:
            chunks_to_transform = self._prepare_chunks_for_split(target_data_to_transform, split, is_multi_unit)
            
            metadata={"mode": split, "apply_to_keys": apply_to_keys, "assign_to_keys":assign_to_keys, "assign_to_map": assign_to_keys_map}
            # Call the appropriate transform method
            if is_multi_unit:
                transformed_chunks = transform_instance.transform_multi_source(chunks_to_transform, metadata=metadata)
            else:
                transformed_chunks = [transform_instance.transform_data(chunks_to_transform[0], metadata=metadata)]
            
            # TODO :(
            transformed_chunks = self._postprocess_transformed_data(transformed_chunks)

            # --- 4. MERGE RESULTS (IN-PLACE OR NEW KEY) --- #
            
            # Mode 1: Assign results to a key defined by assign_to_keys
            for assign_to_key, assign_to_map in zip(assign_to_keys, assign_to_keys_map):
                transformed_results_for_new_key[assign_to_key][split] = [chunk[assign_to_map] for chunk in transformed_chunks]

        
        for assign_to_key, splits in transformed_results_for_new_key.items():
            for split, data in splits.items():
                # Check whenever assign_to_key exist in transformed_data.keys()
                if assign_to_key not in transformed_data:
                    transformed_data[assign_to_key] = {'train':None, 'val':None, 'test':None}

                transformed_data[assign_to_key][split] = transformed_results_for_new_key[assign_to_key][split]

        return transformed_data
    
    def _postprocess_transformed_data(self, data, assign_to_map):
        # Check whenever the transformed data is mapping or not
        if len(data) == 1:
            if isinstance(data, Mapping):
                # TODO: maybe add some checks here
                assert len(metadata['assign_to_map'])==1
                data = SimpleReturnObject(**{metadata['assign_to_map'][0]: data})
        
        return data
            
            
    def _prepare_chunks_for_split(self, data_to_prepare: Dict, split: str, is_multi_unit: bool) -> List['OrderedInput']:
        
        keys = list(data_to_prepare.keys())
    
        # For multi-unit data, extract each unit for each key
        chunks = []
        num_units = len(data_to_prepare[keys[0]][split])
        
        if is_multi_unit == True:
            assert num_units > 1, "Flag 'is_multi_unit' is true hence expect multiple units."

        for i in range(num_units):
            unit_data = {
            key: data_to_prepare[key][split][i]
            for key in keys
            if split in data_to_prepare[key]
            }
            chunks.append(OrderedInput(**unit_data).get_sanitized_data())
        return chunks


# class TransformStrategy():
#     """
#     Abstract that orchestrates how does the transformation is applied to data.
#     """
#     def __init__(self):
#         pass
    
#     def apply(self,
#               transform_instance,
#               data: Dict[str, Any]| DatasetContainer,
#               apply_to_keys: str | List[str],
#               assign_to_key: str, 
#               fit_on_split: str,
#               fit_on_key: str,
#               ) -> Dict[str, Any]:

#         if isinstance(apply_to_keys, list):
#             assert all(isinstance(data[key], dict) for key in apply_to_keys), (
#                 f"Expected a dictionary of train/val/test for apply_to_keys '{apply_to_keys}', "
#                 f"but got {[type(data[key]) for key in apply_to_keys]}"
#             )
#         else:
#             assert isinstance(data[apply_to_keys], dict), f"Expected a dictionary of train/val/test for apply_to_keys '{apply_to_keys}', but got {type(data[apply_to_keys])}"
#             # Convert str to list for uniform processing
#             apply_to_keys = [apply_to_keys]
        
#         # Decouple save copy of our data transformed_data
#         transformed_data = copy.deepcopy(data)
        
    
#         # Collect all key/values for apply_to_keys list
#         target_data_splits = {key: transformed_data[key] for key in apply_to_keys}
        
#         # The target_data_splits structure when apply_to_keys=["features", "timestamps"] is:
#         # in the case of multi-source:
#         # {
#         #     "features": {"train": [...], "val": [...], "test": [...]},
#         #     "timestamps": {"train": [...], "val": [...], "test": [...]},
#         # }
#         # in the case of single-source:
#         # {
#         #     "features": {"train": np.ndarray "val": np.ndarray "test": np.ndarray},
#         #     "timestamps": {"train": np.ndarray "val": np.ndarray "test": np.ndarray},
        
#         # Check that fit_on_split is not None
#         if fit_on_split is not None and fit_on_key is not None:
            
#             # Check that fit_on_split exists in all target_data_splits
#             if any([fit_on_split not in target_data_splits[key].keys() for key in apply_to_keys]):
#                 raise KeyError(f"'{fit_on_split}' not found in data[{apply_to_keys}]")

#             # To stay consistent with the fit_data and transform_data method's inputs
#             fit_data = {}
#             fit_data[fit_on_key] = target_data_splits[fit_on_key][fit_on_split]
            
#             # Seems that single_multi_data_flag will not work with when multiunit has just one unit?  
#             single_multi_data_flag = isinstance(fit_data[fit_on_key], list)
#             if single_multi_data_flag:
#                 transform_instance.fit_multi_source(fit_data, metadata={"mode": fit_on_split})
#             else:
#                 transform_instance.fit_data(fit_data, metadata={"mode": fit_on_split})
#         else: 
#             # TODO: Such flag can fail when we have single source but with list data, need to improve the check
#             # In the case of multisource, data is provided as dictionary of lists
#             single_multi_data_flag = all(
#                 isinstance(v, list)
#                 for inner in target_data_splits.values()
#                 for v in inner.values()
#             )

#         transformed_target_data_splits = {assign_to_key: {}}
#         for split_name in ['train', 'val', 'test']:
            
#             # TODO: handle the case when the target_data_splits[apply_to_keys]['val'] is None for one of apply_to_keys
#             # if split_data is not None:
#             #     pass
#             # else:
#             #     transformed_target_data_splits[split_name] = None
#             #     warnings.warn(f"Split '{split_name}' is None.")
#             # In the case of multisource, data is provided as a list of dictionaries

#             prepare_fn = self._prepare_multi_data if single_multi_data_flag else self._prepare_single_data
#             transform_fn = transform_instance.transform_multi_source if single_multi_data_flag else transform_instance.transform_data

#             prepared_data = prepare_fn(target_data_splits, split_name)
#             transformed_target_data_splits[assign_to_key][split_name] = transform_fn(
#                 prepared_data,
#                 metadata={"mode": split_name}
#             )
            

#         transformed_data[assign_to_key] = transformed_target_data_splits[assign_to_key]
#         return transformed_data  

    


#     def _prepare_single_data(self, data, split_name):
#         aux_data = {}
#         keys = list(data.keys())
    
#         for key in keys:
#             splits = data.get(key, {})
#             if isinstance(splits, dict) and split_name in splits:
#                 aux_data[key] = splits[split_name]
#             else:
#                 warnings.warn(f"Missing or None auxiliary '{key}' for split '{split_name}'.")
#         return aux_data

#     def _prepare_multi_data(self, data, split_name):
#         keys = list(data.keys())
#         aux_data = {}
        
#         for key in keys:
#             splits = data.get(key, {})
#             if split_name not in splits:
#                 warnings.warn(f"Missing auxiliary key '{key}' for split '{split_name}'.")
#                 continue

#             split_aux_list = splits[split_name]
#             if not isinstance(split_aux_list, list):
#                 raise ValueError(
#                     f"Expected a list of items for key '{key}' on split '{split_name}', "
#                     f"but got type {type(split_aux_list)} with length {len(split_aux_list) if isinstance(split_aux_list, list) else 'N/A'}."
#                 )

#             aux_data[key] = split_aux_list

#         return aux_data