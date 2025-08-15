import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np  # Used for stacking in MultiSourceFitTransformStrategy

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
              apply_to_key: str, 
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
        apply_to_key : str
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

    def _prepare_single_aux(self, data, auxiliary_keys, split_name):
        aux_data = {}
        if auxiliary_keys:
            for aux_key in auxiliary_keys:
                aux_splits = data.get(aux_key, {})
                if isinstance(aux_splits, dict) and split_name in aux_splits:
                    aux_data[aux_key] = aux_splits[split_name]
                else:
                    warnings.warn(f"Missing or None auxiliary '{aux_key}' for split '{split_name}'.")
        return aux_data

    def _prepare_multi_aux(self, data, auxiliary_keys, split_name, num_sources):
        if auxiliary_keys:
            missing_keys = [key for key in auxiliary_keys if key not in data]
            if missing_keys:
                raise KeyError(f"Auxiliary keys {missing_keys} not found in data. Available keys: {list(data.keys())}")

        aux_data = {}
        if auxiliary_keys:
            for aux_key in auxiliary_keys:
                aux_splits = data.get(aux_key, {})
                if split_name not in aux_splits:
                    warnings.warn(f"Missing auxiliary key '{aux_key}' for split '{split_name}'.")
                    continue

                split_aux_list = aux_splits[split_name]
                if not isinstance(split_aux_list, list) or len(split_aux_list) != num_sources:
                    raise ValueError(
                        f"Expected a list of {num_sources} items for auxiliary key '{aux_key}' on split '{split_name}', "
                        f"but got type {type(split_aux_list)} with length {len(split_aux_list) if isinstance(split_aux_list, list) else 'N/A'}."
                    )

                aux_data[aux_key] = split_aux_list

        return aux_data

        


class UnifiedDirectTransformStrategy(TransformStrategy):
    """
    Strategy that applies a direct (no fit) transform to single-source or multi-source data.
    """

    def apply(self,
              transform_instance,
              data: Dict[str, Any],
              apply_to_key: str,
              fit_on_split: Optional[str] = None,
              auxiliary_keys: Optional[List[str]] = None
              ) -> Dict[str, Any]:

        assert isinstance(data[apply_to_key], dict), f"Expected a dictionary of train/val/test for apply_to_key '{apply_to_key}', but got {type(data[apply_to_key])}"
        
        
        transformed_data = data.copy()
        target_data_splits = transformed_data[apply_to_key]
        transformed_target_data_splits = {}

        for split_name, split_data in target_data_splits.items():
            if split_data is not None:
                
                # In the case of multisource, data is provided as a list of dictionaries
                if isinstance(split_data, list):
                    aux_data_list = self._prepare_multi_aux(data, auxiliary_keys, split_name, len(split_data))
                    transformed_target_data_splits[split_name] = transform_instance.transform_multi_source(
                        split_data,
                        auxiliary_data=aux_data_list,
                        mode=split_name
                    )
                else:
                    aux_data = self._prepare_single_aux(data, auxiliary_keys, split_name)
                    transformed_target_data_splits[split_name] = transform_instance.transform_data(
                        split_data,
                        auxiliary_data=aux_data,
                        mode=split_name
                    )
            else:
                transformed_target_data_splits[split_name] = None
                warnings.warn(f"Split '{split_name}' is None.")
        
        transformed_data[apply_to_key] = transformed_target_data_splits
        return transformed_data


class UnifiedFitTransformStrategy(TransformStrategy):
    """
    Strategy that fits on one split and applies the transform to all splits,
    for both single-source and multi-source data.
    """

    def apply(self,
              transform_instance,
              data: Dict[str, Any],
              apply_to_key: str,
              fit_on_split: str,
              auxiliary_keys: Optional[List[str]] = None
              ) -> Dict[str, Any]:

        assert isinstance(data[apply_to_key], dict), f"Expected a dictionary for apply_to_key '{apply_to_key}', but got {type(data[apply_to_key])}"
        
        transformed_data = data.copy()
        target_data_splits = transformed_data[apply_to_key]

        if fit_on_split not in target_data_splits:
            raise KeyError(f"'{fit_on_split}' not found in data[{apply_to_key}]")
        
        fit_data = target_data_splits[fit_on_split]
        if isinstance(fit_data, list):
            aux_data_list = self._prepare_multi_aux(data, auxiliary_keys, fit_on_split, len(fit_data))
            transform_instance.fit_multi_source(fit_data, auxiliary_data=aux_data_list, mode=fit_on_split)
        else:
            aux_data = self._prepare_single_aux(data, auxiliary_keys, fit_on_split)
            transform_instance.fit_data(fit_data, auxiliary_data=aux_data, mode=fit_on_split)

        transformed_target_data_splits = {}
        for split_name, split_data in target_data_splits.items():
            if split_data is not None:
                # In the case of multisource, data is provided as a list of dictionaries
                if isinstance(split_data, list):
                    aux_data_list = self._prepare_multi_aux(data, auxiliary_keys, split_name, len(split_data))
                    transformed_target_data_splits[split_name] = transform_instance.transform_multi_source(
                        split_data,
                        auxiliary_data=aux_data_list,
                        mode=split_name
                    )
                else:
                    aux_data = self._prepare_single_aux(data, auxiliary_keys, split_name)
                    transformed_target_data_splits[split_name] = transform_instance.transform_data(
                        split_data,
                        auxiliary_data=aux_data,
                        mode=split_name
                    )
            else:
                transformed_target_data_splits[split_name] = None
                warnings.warn(f"Split '{split_name}' is None.")

        transformed_data[apply_to_key] = transformed_target_data_splits
        return transformed_data       