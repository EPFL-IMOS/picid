from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import warnings
import numpy as np # Used for stacking in MultiSourceFitTransformStrategy
from lmetk.transforms.base_transform import BaseTransform # Import your BaseTransform ABC

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

        assert isinstance(data[apply_to_key], dict), f"Expected a dictionary for apply_to_key '{apply_to_key}', but got {type(data[apply_to_key])}"
        

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




# --- Transform Strategies ---
# class TransformStrategy(ABC):
#     """
#     Abstract base class for different transformation strategies.
#     Each strategy defines how a transformation instance is applied to data.
#     """
    
    # @abstractmethod
    # def apply(self, 
    #           transform_instance: BaseTransform, 
    #           data: Union[Dict[str, Any], List[Dict[str, Any]]], 
    #           apply_to_key: str, 
    #           fit_on_split: Optional[str] = None
    #          ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    #     """
    #     Applies the transform strategy to the data.

    #     Parameters
    #     ----------
    #     transform_instance : BaseTransform
    #         The actual transformation object, which must implement BaseTransform.
    #     data : Union[Dict[str, Any], List[Dict[str, Any]]]
    #         The input data. Can be a single dictionary representing one source's
    #         splits, or a list of such dictionaries for multiple sources.
    #     apply_to_key : str
    #         The top-level key within each source's data dictionary (e.g., 'features')
    #         to which the transformation should be applied.
    #     fit_on_split : Optional[str], default=None
    #         The name of the data split (e.g., 'train') to use for fitting the
    #         transform, if required by the strategy.

    #     Returns
    #     -------
    #     Union[Dict[str, Any], List[Dict[str, Any]]]
    #         The transformed data, in the same structure as the input data.
    #     """
    #     pass


# class DirectTransformStrategy(TransformStrategy):
#     """
#     Strategy for transforms that are applied directly to each split
#     without requiring a fitting step. Operates on a single source's data.
#     """
    
#     def apply(self, 
#               transform_instance: BaseTransform, 
#               data: Dict[str, Any], 
#               apply_to_key: str, 
#               auxiliary_keys: Optional[List[str]] = None,
#               fit_on_split: Optional[str] = None # Not used by this strategy, but kept for signature consistency
#              ) -> Dict[str, Any]:
#         """
#         Applies the transform directly to the specified `apply_to_key` within
#         each split of the single source's data, using `transform_instance.transform_data()`.
        
#         Args:
#             transform_instance: The transform to apply
#             data: Dictionary containing the data splits
#             apply_to_key: Key identifying which data to transform
#             auxiliary_keys: Optional list of keys for auxiliary information to pass to transform
#             fit_on_split: Not used by this strategy, kept for signature consistency
            
#         Returns:
#             Dictionary with transformed data
#         """
#         if not isinstance(data, dict):
#             raise TypeError(f"{self.__class__.__name__} expects a single data dictionary (for a single source).")
#         if apply_to_key not in data:
#             raise KeyError(f"Data key '{apply_to_key}' not found in data. Available keys: {list(data.keys())}")
        
#         # Validate auxiliary keys if provided
#         if auxiliary_keys:
#             missing_keys = [key for key in auxiliary_keys if key not in data]
#             if missing_keys:
#                 raise KeyError(f"Auxiliary keys {missing_keys} not found in data. Available keys: {list(data.keys())}")
        
#         # Make a copy to avoid modifying original input data directly
#         transformed_data = data.copy()
        
#         target_data_splits = transformed_data[apply_to_key]
#         transformed_target_data_splits = {}
        
#         for split_name, split_data in target_data_splits.items():
#             if split_data is not None:
#                 # Prepare auxiliary data for this split if auxiliary_keys are provided
#                 auxiliary_data = {}
                # if auxiliary_keys:
                #     for aux_key in auxiliary_keys:
                #         aux_splits = data[aux_key]
                #         if split_name in aux_splits and aux_splits[split_name] is not None:
                #             auxiliary_data[aux_key] = aux_splits[split_name]
                #         else:
                #             warnings.warn(f"Auxiliary key '{aux_key}' for split '{split_name}' is None or missing.")
                
#                 # Apply transform with or without auxiliary data
#                 #if auxiliary_data:
#                 # Assuming transform_instance.transform_data can accept auxiliary data
#                 transformed_target_data_splits[split_name] = transform_instance.transform_data(
#                     split_data, 
#                     auxiliary_data=auxiliary_data,
#                     mode=split_name,
#                 )
#                 # else:
#                 #     transformed_target_data_splits[split_name] = transform_instance.transform_data(split_data, mode=split_name)
#             else:
#                 transformed_target_data_splits[split_name] = None
#                 warnings.warn(f"For feature '{apply_to_key}', the split '{split_name}' is None.")
        
#         transformed_data[apply_to_key] = transformed_target_data_splits
        
#         return transformed_data


# class FitTransformStrategy(TransformStrategy):
#     """
#     Strategy for transforms that need to be fitted on a specific split
#     (e.g., 'train') of a single source, and then applied to all splits
#     within that same source.
#     """
    
#     def apply(self, 
#               transform_instance: BaseTransform, 
#               data: Dict[str, Any], 
#               apply_to_key: str, 
#               fit_on_split: str,
#               auxiliary_keys: Optional[List[str]] = None
#              ) -> Dict[str, Any]:
#         """
#         Fits the transform instance on the specified `fit_on_split` data
#         from `apply_to_key` of a single source, then transforms all splits
#         under the same `apply_to_key`. Uses `transform_instance.fit_data()`
#         and `transform_instance.transform_data()`.
        
#         Args:
#             transform_instance: The transform to apply
#             data: Dictionary containing the data splits
#             apply_to_key: Key identifying which data to transform
#             fit_on_split: Split to use for fitting the transform
#             auxiliary_keys: Optional list of keys for auxiliary information to pass to transform
            
#         Returns:
#             Dictionary with transformed data
#         """
#         if not isinstance(data, dict):
#             raise TypeError(f"{self.__class__.__name__} expects a single data dictionary (for a single source).")
#         if apply_to_key not in data:
#             raise KeyError(f"Data key '{apply_to_key}' not found in data. Available keys: {list(data.keys())}")
#         if fit_on_split is None:
#             raise ValueError(f"{self.__class__.__name__} requires 'fit_on_split' to be specified.")
        
#         # Validate auxiliary keys if provided
#         if auxiliary_keys:
#             missing_keys = [key for key in auxiliary_keys if key not in data]
#             if missing_keys:
#                 raise KeyError(f"Auxiliary keys {missing_keys} not found in data. Available keys: {list(data.keys())}")
        
#         # Make a copy to avoid modifying original input data directly
#         transformed_data = data.copy()

#         target_data_splits = transformed_data[apply_to_key]

#         # 1. Fit on the specified split using BaseTransform's single-source fit method
#         if fit_on_split in target_data_splits:
#             try:
#                 # Prepare auxiliary data for fitting if auxiliary_keys are provided
#                 fit_auxiliary_data = {}
#                 if auxiliary_keys:
#                     for aux_key in auxiliary_keys:
#                         aux_splits = data[aux_key]
#                         if fit_on_split in aux_splits and aux_splits[fit_on_split] is not None:
#                             fit_auxiliary_data[aux_key] = aux_splits[fit_on_split]
#                         else:
#                             warnings.warn(f"Auxiliary key '{aux_key}' for fitting split '{fit_on_split}' is None or missing.")
                
#                 # Assuming fit_data can accept auxiliary data
#                 transform_instance.fit_data(
#                     target_data_splits[fit_on_split], 
#                     auxiliary_data=fit_auxiliary_data,
#                     mode=fit_on_split,
#                 )
                
                    
#             except Exception as e:
#                 raise RuntimeError(f"Error during single-source fitting of '{transform_instance.__class__.__name__}' on split '{fit_on_split}': {e}")
#         else:
#             raise KeyError(f"Split '{fit_on_split}' not found under key '{apply_to_key}' in data. Available splits: {list(target_data_splits.keys())}")
        
#         # 2. Transform all splits using BaseTransform's transform method
#         transformed_target_data_splits = {}
#         for split_name, split_data in target_data_splits.items():
#             if split_data is not None:
#                 # Prepare auxiliary data for this split if auxiliary_keys are provided
#                 auxiliary_data = {}
#                 if auxiliary_keys:
#                     for aux_key in auxiliary_keys:
#                         aux_splits = data[aux_key]
#                         if split_name in aux_splits and aux_splits[split_name] is not None:
#                             auxiliary_data[aux_key] = aux_splits[split_name]
#                         else:
#                             warnings.warn(f"Auxiliary key '{aux_key}' for split '{split_name}' is None or missing.")
                
#                 # Apply transform with or without auxiliary data
#                 #if auxiliary_data:
#                 transformed_target_data_splits[split_name] = transform_instance.transform_data(
#                     split_data, 
#                     auxiliary_data=auxiliary_data,
#                     mode=split_name,
#                 )
#                 # else:
#                 #     transformed_target_data_splits[split_name] = transform_instance.transform_data(split_data, mode=split_name)
#             else:
#                 transformed_target_data_splits[split_name] = None
#                 warnings.warn(f"For feature '{apply_to_key}', the split '{split_name}' is None.")
                
        
#         transformed_data[apply_to_key] = transformed_target_data_splits
        
#         return transformed_data


# class MultiSourceFitTransformStrategy(TransformStrategy):
#     """
#     Strategy for transforms that need to be fitted across multiple sources (datasets)
#     on a specified split (e.g., 'train'), and then applied to corresponding splits
#     within each source separately. Uses `transform_instance.fit_multi_source()`
#     and `transform_instance.transform_data()`.
#     """
    
#     def apply(self, 
#               transform_instance: BaseTransform, 
#               data: Dict[str, Any], 
#               apply_to_key: str, 
#               fit_on_split: str,
#               auxiliary_keys: Optional[List[str]] = None
#              ) -> List[Dict[str, Any]]:
#         """
#         Fits the transform instance on aggregated data from the `fit_on_split`
#         across all sources (from `apply_to_key`), then applies the fitted
#         transform to the `apply_to_key` within each split of each source.
        
#         Args:
#             transform_instance: The transform to apply
#             list_of_data_dicts: List of data dictionaries (one per source)
#             apply_to_key: Key identifying which data to transform
#             fit_on_split: Split to use for fitting the transform
#             auxiliary_keys: Optional list of keys for auxiliary information to pass to transform
            
#         Returns:
#             List of dictionaries with transformed data
#         """
#         # Checking the inputs
#         for d in data.values():
#             for split in ['train', 'val', 'test']:
#                 if d[split]:
#                     if not isinstance(d[split], list):
#                         raise TypeError(f"Data for split '{split}' must be a list.")
        
        
#         #TODO: The logic belo needs to be adapted
        
#         # Make a copy to avoid modifying original input data directly
#         transformed_data = data.copy()

#         target_data_splits = transformed_data[apply_to_key]

#         # 1. Fit on the specified split using BaseTransform's single-source fit method
#         if fit_on_split in target_data_splits:
#             try:
#                 # Prepare auxiliary data for fitting if auxiliary_keys are provided
#                 fit_auxiliary_data = {}
#                 if auxiliary_keys:
#                     for aux_key in auxiliary_keys:
#                         aux_splits = data[aux_key]
#                         if fit_on_split in aux_splits and aux_splits[fit_on_split] is not None:
#                             fit_auxiliary_data[aux_key] = aux_splits[fit_on_split]
#                         else:
#                             warnings.warn(f"Auxiliary key '{aux_key}' for fitting split '{fit_on_split}' is None or missing.")
                
#                 # Fit with or without auxiliary data
#                 # if fit_auxiliary_data:
#                     # Assuming fit_data can accept auxiliary data
#                 transform_instance.fit_multi_source(
#                     data_segments=target_data_splits[fit_on_split], 
#                     auxiliary_data=fit_auxiliary_data,
#                     mode=fit_on_split
#                 )
#                 # else:
#                 #     transform_instance.fit_data(data_segments=target_data_splits[fit_on_split], mode=fit_on_split, auxiliary_data=fit_auxiliary_data,)
                    
#             except Exception as e:
#                 raise RuntimeError(f"Error during single-source fitting of '{transform_instance.__class__.__name__}' on split '{fit_on_split}': {e}")
#         else:
#             raise KeyError(f"Split '{fit_on_split}' not found under key '{apply_to_key}' in data. Available splits: {list(target_data_splits.keys())}")
        
#         # 2. Transform all splits using BaseTransform's transform method
#         transformed_target_data_splits = {}
#         for split_name, split_data in target_data_splits.items():
#             if split_data is not None:
#                 # Prepare auxiliary data for this split if auxiliary_keys are provided
#                 auxiliary_data = {}
#                 if auxiliary_keys:
#                     for aux_key in auxiliary_keys:
#                         aux_splits = data[aux_key]
#                         if split_name in aux_splits and aux_splits[split_name] is not None:
#                             auxiliary_data[aux_key] = aux_splits[split_name]
#                         else:
#                             warnings.warn(f"Auxiliary key '{aux_key}' for split '{split_name}' is None or missing.")
                
#                 # Apply transform with or without auxiliary data
#                 #if auxiliary_data:
#                 transformed_target_data_splits[split_name] = transform_instance.transform_multi_source(
#                     data_segments=split_data, 
#                     auxiliary_data=auxiliary_data,
#                     mode=split_name,
#                 )
#                 # else:
#                 #     transformed_target_data_splits[split_name] = transform_instance.transform_data(split_data, mode=split_name)
#             else:
#                 transformed_target_data_splits[split_name] = None
#                 warnings.warn(f"For feature '{apply_to_key}', the split '{split_name}' is None.")
                
#         # Update the data dict  with the transformed data splits 
#         assert all([key in transformed_target_data_splits.keys() for key in ['train', 'val', 'test']])
#         transformed_data[apply_to_key] = transformed_target_data_splits
        
#         return transformed_data

# class MultiSourceDirectTransformStrategy(TransformStrategy):
#     """
#     Strategy for transforms that need to be fitted across multiple sources (datasets)
#     on a specified split (e.g., 'train'), and then applied to corresponding splits
#     within each source separately. Uses `transform_instance.fit_multi_source()`
#     and `transform_instance.transform_data()`.
#     """
    
#     def apply(self, 
#               transform_instance: BaseTransform, 
#               data: Dict[str, Any], 
#               apply_to_key: str, 
#               fit_on_split: str,
#               auxiliary_keys: Optional[List[str]] = None
#              ) -> List[Dict[str, Any]]:
#         """
#         Fits the transform instance on aggregated data from the `fit_on_split`
#         across all sources (from `apply_to_key`), then applies the fitted
#         transform to the `apply_to_key` within each split of each source.
        
#         Args:
#             transform_instance: The transform to apply
#             list_of_data_dicts: List of data dictionaries (one per source)
#             apply_to_key: Key identifying which data to transform
#             fit_on_split: Split to use for fitting the transform
#             auxiliary_keys: Optional list of keys for auxiliary information to pass to transform
            
#         Returns:
#             List of dictionaries with transformed data
#         """
#         # Checking the inputs
#         for d in data.values():
#             for split in ['train', 'val', 'test']:
#                 if d[split]:
#                     if not isinstance(d[split], list):
#                         raise TypeError(f"Data for split '{split}' must be a list.")
        
        
#         # Validate auxiliary keys if provided
#         if auxiliary_keys:
#             missing_keys = [key for key in auxiliary_keys if key not in data]
#             if missing_keys:
#                 raise KeyError(f"Auxiliary keys {missing_keys} not found in data. Available keys: {list(data.keys())}")
        
#         # Make a copy to avoid modifying original input data directly
#         transformed_data = data.copy()
        
#         target_data_splits = transformed_data[apply_to_key]
#         transformed_target_data_splits = {}
        
#         for split_name, split_data in target_data_splits.items():
#             if split_data is not None:
#                 # Prepare auxiliary data for this split if auxiliary_keys are provided
#                 auxiliary_data = {}
#                 if auxiliary_keys:
#                     for aux_key in auxiliary_keys:
#                         aux_splits = data[aux_key]
#                         if split_name in aux_splits and aux_splits[split_name] is not None:
#                             auxiliary_data[aux_key] = aux_splits[split_name]
#                         else:
#                             warnings.warn(f"Auxiliary key '{aux_key}' for split '{split_name}' is None or missing.")
                
#                 # Apply transform with or without auxiliary data
#                 #if auxiliary_data:
#                     # Assuming transform_instance.transform_data can accept auxiliary data
#                 transformed_target_data_splits[split_name] = transform_instance.transform_multi_source(
#                     split_data, 
#                     auxiliary_data=auxiliary_data,
#                     mode=split_name,
#                 )
#                 # else:
#                 #     transformed_target_data_splits[split_name] = transform_instance.transform_multi_source(split_data, mode=split_name)
#             else:
#                 transformed_target_data_splits[split_name] = None
#                 warnings.warn(f"For feature '{apply_to_key}', the split '{split_name}' is None.")
        
#         transformed_data[apply_to_key] = transformed_target_data_splits
        
#         return transformed_data