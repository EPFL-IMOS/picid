import warnings
from typing import Any, Dict, List, Optional, Union
import copy



class TransformStrategy():
    """
    Abstract base class for different transformation strategies.
    Each strategy defines how a transformation instance is applied to data.
    """
    def __init__(self):
        pass
    
    def apply(self,
              transform_instance,
              data: Dict[str, Any],
              apply_to_keys: str | List[str],
              assign_to_key: str, 
              fit_on_split: str,
              fit_on_key: str,
              ) -> Dict[str, Any]:

        if isinstance(apply_to_keys, list):
            assert all(isinstance(data[key], dict) for key in apply_to_keys), (
                f"Expected a dictionary of train/val/test for apply_to_keys '{apply_to_keys}', "
                f"but got {[type(data[key]) for key in apply_to_keys]}"
            )
        else:
            assert isinstance(data[apply_to_keys], dict), f"Expected a dictionary of train/val/test for apply_to_keys '{apply_to_keys}', but got {type(data[apply_to_keys])}"
            # Convert str to list for uniform processing
            apply_to_keys = [apply_to_keys]
        
        # Decouple save copy of our data transformed_data
        transformed_data = copy.deepcopy(data)
        
    
        # Collect all key/values for apply_to_keys list
        target_data_splits = {key: transformed_data[key] for key in apply_to_keys}
        
        # The target_data_splits structure when apply_to_keys=["features", "timestamps"] is:
        # in the case of multi-source:
        # {
        #     "features": {"train": [...], "val": [...], "test": [...]},
        #     "timestamps": {"train": [...], "val": [...], "test": [...]},
        # }
        # in the case of single-source:
        # {
        #     "features": {"train": np.ndarray "val": np.ndarray "test": np.ndarray},
        #     "timestamps": {"train": np.ndarray "val": np.ndarray "test": np.ndarray},
        
        # Check that fit_on_split is not None
        if fit_on_split is not None and fit_on_key is not None:
            
            # Check that fit_on_split exists in all target_data_splits
            if any([fit_on_split not in target_data_splits[key].keys() for key in apply_to_keys]):
                raise KeyError(f"'{fit_on_split}' not found in data[{apply_to_keys}]")

            # To stay consistent with the fit_data and transform_data method's inputs
            fit_data = {}
            fit_data[fit_on_key] = target_data_splits[fit_on_key][fit_on_split]
            
            # Seems that single_multi_data_flag will not work with when multiunit has just one unit?  
            single_multi_data_flag = isinstance(fit_data[fit_on_key], list)
            if single_multi_data_flag:
                transform_instance.fit_multi_source(fit_data, metadata={"mode": fit_on_split})
            else:
                transform_instance.fit_data(fit_data, metadata={"mode": fit_on_split})
        else: 
            # TODO: Such flag can fail when we have single source but with list data, need to improve the check
            # In the case of multisource, data is provided as dictionary of lists
            single_multi_data_flag = all(
                isinstance(v, list)
                for inner in target_data_splits.values()
                for v in inner.values()
            )

        transformed_target_data_splits = {assign_to_key: {}}
        for split_name in ['train', 'val', 'test']:
            
            # TODO: handle the case when the target_data_splits[apply_to_keys]['val'] is None for one of apply_to_keys
            # if split_data is not None:
            #     pass
            # else:
            #     transformed_target_data_splits[split_name] = None
            #     warnings.warn(f"Split '{split_name}' is None.")
            # In the case of multisource, data is provided as a list of dictionaries

            prepare_fn = self._prepare_multi_data if single_multi_data_flag else self._prepare_single_data
            transform_fn = transform_instance.transform_multi_source if single_multi_data_flag else transform_instance.transform_data

            prepared_data = prepare_fn(target_data_splits, split_name)
            transformed_target_data_splits[assign_to_key][split_name] = transform_fn(
                prepared_data,
                metadata={"mode": split_name}
            )
            

        transformed_data[assign_to_key] = transformed_target_data_splits[assign_to_key]
        return transformed_data  

    


    def _prepare_single_data(self, data, split_name):
        aux_data = {}
        keys = list(data.keys())
    
        for key in keys:
            splits = data.get(key, {})
            if isinstance(splits, dict) and split_name in splits:
                aux_data[key] = splits[split_name]
            else:
                warnings.warn(f"Missing or None auxiliary '{key}' for split '{split_name}'.")
        return aux_data

    def _prepare_multi_data(self, data, split_name):
        keys = list(data.keys())
        aux_data = {}
        
        for key in keys:
            splits = data.get(key, {})
            if split_name not in splits:
                warnings.warn(f"Missing auxiliary key '{key}' for split '{split_name}'.")
                continue

            split_aux_list = splits[split_name]
            if not isinstance(split_aux_list, list):
                raise ValueError(
                    f"Expected a list of items for key '{key}' on split '{split_name}', "
                    f"but got type {type(split_aux_list)} with length {len(split_aux_list) if isinstance(split_aux_list, list) else 'N/A'}."
                )

            aux_data[key] = split_aux_list

        return aux_data