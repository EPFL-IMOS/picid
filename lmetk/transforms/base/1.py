import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import omegaconf
import numpy as np  # Used for stacking in MultiSourceFitTransformStrategy
import copy
from lmetk.transforms.base.base_transform import \
    BaseTransform  # Import your BaseTransform ABC


class UnifiedDirectTransformStrategy(TransformStrategy):
    """
    Strategy that applies a direct (no fit) transform to single-source or multi-source data.
    """

    def apply(self,
              transform_instance,
              data: Dict[str, Any],
              apply_to_keys: str | List[str],
              assign_to_key: str, 
              **kwargs
              ) -> Dict[str, Any]:
        
        if isinstance(apply_to_keys, omegaconf.listconfig.ListConfig):
            apply_to_keys = list(apply_to_keys)

        if isinstance(apply_to_keys, list):
            assert all(isinstance(data[key], dict) for key in apply_to_keys), (
                f"Expected a dictionary of train/val/test for apply_to_keys '{apply_to_keys}', "
                f"but got {[type(data[key]) for key in apply_to_keys]}"
            )
            
        else:
            assert isinstance(data[apply_to_keys], dict), f"Expected a dictionary of train/val/test for apply_to_keys '{apply_to_keys}', but got {type(data[apply_to_keys])}"
            apply_to_keys = [apply_to_keys]
        
        
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
        
        transformed_target_data_splits = {assign_to_key: {}}
        for split_name in ['train', 'val', 'test']:
            # TODO: handle the case when the target_data_splits[apply_to_keys]['val'] is None for one of apply_to_keys
            # if split_data is not None:
            #     pass
            # else:
            #     transformed_target_data_splits[split_name] = None
            #     warnings.warn(f"Split '{split_name}' is None.")
                
            
            
            # TODO: Such flag can fail when we have single source but with list data, need to improve the check
            # In the case of multisource, data is provided as dictionary of lists
            single_multi_data_flag = all(
                isinstance(v, list)
                for inner in target_data_splits.values()
                for v in inner.values()
            )
            if single_multi_data_flag:
                prepared_data = self._prepare_multi_data(target_data_splits, split_name)
                transformed_target_data_splits[assign_to_key][split_name] = transform_instance.transform_multi_source(
                    prepared_data,
                    metadata={"mode": split_name},
                )
            else:
                prepared_data = self._prepare_single_data(target_data_splits, split_name)
                transformed_target_data_splits[assign_to_key][split_name] = transform_instance.transform_data(
                    prepared_data, 
                    metadata={"mode": split_name},
                )
            
        
        transformed_data[assign_to_key] = transformed_target_data_splits[assign_to_key]
        return transformed_data