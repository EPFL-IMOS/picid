from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np # Used for stacking in MultiSourceFitTransformStrategy
from lmetk.transforms.base_transform import BaseTransform # Import your BaseTransform ABC

# --- Transform Strategies ---

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


class DirectTransformStrategy(TransformStrategy):
    """
    Strategy for transforms that are applied directly to each split
    without requiring a fitting step. Operates on a single source's data.
    """
    
    def apply(self, 
              transform_instance: BaseTransform, 
              data: Dict[str, Any], 
              apply_to_key: str, 
              fit_on_split: Optional[str] = None # Not used by this strategy, but kept for signature consistency
             ) -> Dict[str, Any]:
        """
        Applies the transform directly to the specified `apply_to_key` within
        each split of the single source's data, using `transform_instance.transform_data()`.
        """
        if not isinstance(data, dict):
            raise TypeError(f"{self.__class__.__name__} expects a single data dictionary (for a single source).")
        if apply_to_key not in data:
            raise KeyError(f"Data key '{apply_to_key}' not found in data. Available keys: {list(data.keys())}")
        
        # Make a copy to avoid modifying original input data directly
        transformed_data = data.copy()
        
        target_data_splits = transformed_data[apply_to_key]
        transformed_target_data_splits = {}
        
        for split_name, split_data in target_data_splits.items():
            transformed_target_data_splits[split_name] = transform_instance.transform_data(split_data)
        
        transformed_data[apply_to_key] = transformed_target_data_splits
        
        return transformed_data


class FitTransformStrategy(TransformStrategy):
    """
    Strategy for transforms that need to be fitted on a specific split
    (e.g., 'train') of a single source, and then applied to all splits
    within that same source.
    """
    
    def apply(self, 
              transform_instance: BaseTransform, 
              data: Dict[str, Any], 
              apply_to_key: str, 
              fit_on_split: str
             ) -> Dict[str, Any]:
        """
        Fits the transform instance on the specified `fit_on_split` data
        from `apply_to_key` of a single source, then transforms all splits
        under the same `apply_to_key`. Uses `transform_instance.fit_single_source()`
        and `transform_instance.transform_data()`.
        """
        if not isinstance(data, dict):
            raise TypeError(f"{self.__class__.__name__} expects a single data dictionary (for a single source).")
        if apply_to_key not in data:
            raise KeyError(f"Data key '{apply_to_key}' not found in data. Available keys: {list(data.keys())}")
        if fit_on_split is None:
            raise ValueError(f"{self.__class__.__name__} requires 'fit_on_split' to be specified.")
        
        # Make a copy to avoid modifying original input data directly
        transformed_data = data.copy()

        target_data_splits = transformed_data[apply_to_key]

        # 1. Fit on the specified split using BaseTransform's single-source fit method
        if fit_on_split in target_data_splits:
            try:
                transform_instance.fit_single_source(target_data_splits[fit_on_split])
            except Exception as e:
                raise RuntimeError(f"Error during single-source fitting of '{transform_instance.__class__.__name__}' on split '{fit_on_split}': {e}")
        else:
            raise KeyError(f"Split '{fit_on_split}' not found under key '{apply_to_key}' in data. Available splits: {list(target_data_splits.keys())}")
        
        # 2. Transform all splits using BaseTransform's transform method
        transformed_target_data_splits = {}
        for split_name, split_data in target_data_splits.items():
            transformed_target_data_splits[split_name] = transform_instance.transform_data(split_data)
        
        transformed_data[apply_to_key] = transformed_target_data_splits
        
        return transformed_data


class MultiSourceFitTransformStrategy(TransformStrategy):
    """
    Strategy for transforms that need to be fitted across multiple sources (datasets)
    on a specified split (e.g., 'train'), and then applied to corresponding splits
    within each source separately. Uses `transform_instance.fit_multi_source()`
    and `transform_instance.transform_data()`.
    """
    
    def apply(self, 
              transform_instance: BaseTransform, 
              list_of_data_dicts: List[Dict[str, Any]], 
              apply_to_key: str, 
              fit_on_split: str
             ) -> List[Dict[str, Any]]:
        """
        Fits the transform instance on aggregated data from the `fit_on_split`
        across all sources (from `apply_to_key`), then applies the fitted
        transform to the `apply_to_key` within each split of each source.
        """
        if not isinstance(list_of_data_dicts, list) or not all(isinstance(d, dict) for d in list_of_data_dicts):
            raise TypeError(f"{self.__class__.__name__} expects a list of data dictionaries.")
        if fit_on_split is None:
            raise ValueError(f"{self.__class__.__name__} requires 'fit_on_split' to be specified.")

        # 1. Collect data segments for fitting, delegating aggregation to transform_instance
        data_segments_for_fitting = []
        for i, source_data in enumerate(list_of_data_dicts):
            if apply_to_key not in source_data:
                raise KeyError(f"Key '{apply_to_key}' not found in source data dictionary at index {i}.")
            
            if fit_on_split not in source_data[apply_to_key]:
                 raise KeyError(f"Split '{fit_on_split}' not found under '{apply_to_key}' in source data dictionary at index {i} for fitting.")

            data_segments_for_fitting.append(source_data[apply_to_key][fit_on_split])
        
        if not data_segments_for_fitting:
            raise ValueError("No data segments found to fit the multi-source transformer.")

        # 2. Fit the transform instance using its multi-source fit method
        try:
            transform_instance.fit_multi_source(data_segments_for_fitting)
        except Exception as e:
            raise RuntimeError(f"Error during multi-source fitting of '{transform_instance.__class__.__name__}': {e}")

        # 3. Apply the fitted transform to each source's data
        transformed_list_of_data_dicts = []
        for source_data in list_of_data_dicts:
            # Make a shallow copy of the source_data dict itself.
            # The nested dict for apply_to_key will be recreated or updated.
            transformed_source_data = source_data.copy() 

            if apply_to_key in transformed_source_data:
                transformed_splits = {}
                for split_name, split_data in transformed_source_data[apply_to_key].items():
                    # Apply transform to each split using BaseTransform's transform method
                    transformed_splits[split_name] = transform_instance.transform_data(split_data)
                transformed_source_data[apply_to_key] = transformed_splits
            
            transformed_list_of_data_dicts.append(transformed_source_data)
        
        return transformed_list_of_data_dicts