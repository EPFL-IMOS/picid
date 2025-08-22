import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import omegaconf
import numpy as np  # Used for stacking in MultiSourceFitTransformStrategy
import copy
from lmetk.transforms.base.base_transform import \
    BaseTransform  # Import your BaseTransform ABC


class TransformStrategy(ABC):
    """
    Abstract base class for different transformation strategies.
    Each strategy defines how a transformation instance is applied to data.
    """
    
    @abstractmethod
    def apply(self, 
              transform_instance: BaseTransform, 
              data: Union[Dict[str, Any], List[Dict[str, Any]]], 
              apply_to_key: str | List[str],
              fit_on_split: Optional[str] = None
             ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Applies the transform strategy to the data.

        Parameters
        ----------
        transform_instance : BaseTransform
            The actual transformation object, which must implement BaseTransform.
        data : Union[Dict[str, Any], List[Dict[str, Any]]]
            The input data. Can be a single dictionary representing one source's
            splits, or a list of such dictionaries for multiple sources.
        apply_to_keys : str
            The top-level key within each source's data dictionary (e.g., 'features')
            to which the transformation should be applied.
        fit_on_split : Optional[str], default=None
            The name of the data split (e.g., 'train') to use for fitting the
            transform, if required by the strategy.

        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            The transformed data, in the same structure as the input data.
        """
        pass

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


class UnifiedFitTransformStrategy(TransformStrategy):
    """
    Strategy that fits on one split and applies the transform to all splits,
    for both single-source and multi-source data.
    """

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
        
        # Check that fit_on_split exists in all target_data_splits
        if any([fit_on_split not in transformed_data[key].keys() for key in apply_to_keys]):
            raise KeyError(f"'{fit_on_split}' not found in data[{apply_to_keys}]")

        # To stay consistent with the fit_data and transform_data method's inputs
        fit_data = {}
        fit_data[fit_on_key] = target_data_splits[fit_on_key][fit_on_split]
        
        single_multi_data_flag = isinstance(fit_data[fit_on_key], list)
        if single_multi_data_flag:
            transform_instance.fit_multi_source(fit_data, metadata={"mode": fit_on_split})
        else:
            transform_instance.fit_data(fit_data, metadata={"mode": fit_on_split})

        transformed_target_data_splits = {assign_to_key: {}}
        for split_name in ['train', 'val', 'test']:
            
            # TODO: handle the case when the target_data_splits[apply_to_keys]['val'] is None for one of apply_to_keys
            # if split_data is not None:
            #     pass
            # else:
            #     transformed_target_data_splits[split_name] = None
            #     warnings.warn(f"Split '{split_name}' is None.")
            # In the case of multisource, data is provided as a list of dictionaries

            if single_multi_data_flag:
                prepared_data = self._prepare_multi_data(target_data_splits, split_name)
                transformed_target_data_splits[assign_to_key][split_name] = transform_instance.transform_multi_source(
                    prepared_data,
                    metadata={"mode": split_name})
            else:
                prepared_data = self._prepare_single_data(target_data_splits, split_name)
                transformed_target_data_splits[assign_to_key][split_name] = transform_instance.transform_data(
                    prepared_data,
                    metadata={"mode": split_name})
            

        transformed_data[assign_to_key] = transformed_target_data_splits[assign_to_key]
        return transformed_data       