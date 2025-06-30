from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from hydra.utils import instantiate
from omegaconf import DictConfig

# ============================================================================
# SIMPLIFIED BASE TRANSFORM INTERFACE
# ============================================================================

class BaseTransform(ABC):
    """
    Simplified base transform interface. Transforms only need to implement
    the core transformation logic, not the fitting strategies.
    """
    
    @abstractmethod
    def transform_data(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None, 
                      mode: Optional[str] = None) -> Any:
        """
        Transform a single data segment.
        
        Parameters
        ----------
        data : Any
            The data to transform
        auxiliary_data : Optional[Dict[str, Any]]
            Additional data needed for transformation
        mode : Optional[str]
            Transformation mode (e.g., 'train', 'val', 'test')
        """
        pass
    
    def fit_data(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None) -> 'BaseTransform':
        """
        Fit the transform on data. Default implementation does nothing (for stateless transforms).
        Override this method for transforms that need fitting.
        """
        return self
    
    def supports_multi_source_fit(self) -> bool:
        """
        Whether this transform can handle multi-source fitting.
        Override to return True if the transform can handle aggregated data from multiple sources.
        """
        return True  # Most transforms can handle concatenated data

# ============================================================================
# COMPOSITION-BASED FITTING STRATEGIES
# ============================================================================

class FittingStrategy(ABC):
    """Base class for fitting strategies that can be composed with transforms."""
    
    @abstractmethod
    def fit_transform(self, 
                     transform: BaseTransform,
                     data: Union[Dict[str, Any], List[Dict[str, Any]]],
                     apply_to_key: str,
                     auxiliary_keys: Optional[List[str]] = None,
                     **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Apply the fitting strategy."""
        pass

class DirectFittingStrategy(FittingStrategy):
    """Strategy for transforms that don't require fitting."""
    
    def fit_transform(self, 
                     transform: BaseTransform,
                     data: Dict[str, Any],
                     apply_to_key: str,
                     auxiliary_keys: Optional[List[str]] = None,
                     **kwargs) -> Dict[str, Any]:
        
        if not isinstance(data, dict):
            raise TypeError("DirectFittingStrategy expects a single data dictionary")
        
        return self._apply_transform_to_splits(transform, data, apply_to_key, auxiliary_keys)
    
    def _apply_transform_to_splits(self, transform: BaseTransform, data: Dict[str, Any], 
                                  apply_to_key: str, auxiliary_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Apply transform to all splits in the data."""
        result = data.copy()
        target_splits = result[apply_to_key]
        transformed_splits = {}
        
        for split_name, split_data in target_splits.items():
            if split_data is not None:
                # Prepare auxiliary data for this split
                auxiliary_data = {}
                if auxiliary_keys:
                    for aux_key in auxiliary_keys:
                        if aux_key in data and split_name in data[aux_key]:
                            auxiliary_data[aux_key] = data[aux_key][split_name]
                        else:
                            warnings.warn(f"Auxiliary key '{aux_key}' for split '{split_name}' not found")
                
                transformed_splits[split_name] = transform.transform_data(
                    split_data, 
                    auxiliary_data=auxiliary_data if auxiliary_data else None,
                    mode=split_name
                )
            else:
                transformed_splits[split_name] = None
                warnings.warn(f"Split '{split_name}' is None for key '{apply_to_key}'")
        
        result[apply_to_key] = transformed_splits
        return result

class SingleSourceFittingStrategy(FittingStrategy):
    """Strategy for transforms that fit on one split of a single source."""
    
    def fit_transform(self, 
                     transform: BaseTransform,
                     data: Dict[str, Any],
                     apply_to_key: str,
                     fit_on_split: str,
                     auxiliary_keys: Optional[List[str]] = None,
                     **kwargs) -> Dict[str, Any]:
        
        if not isinstance(data, dict):
            raise TypeError("SingleSourceFittingStrategy expects a single data dictionary")
        
        if fit_on_split is None:
            raise ValueError("fit_on_split must be specified")
        
        # Fit on the specified split
        fit_data = data[apply_to_key][fit_on_split]
        
        # Prepare auxiliary data for fitting
        fit_auxiliary_data = {}
        if auxiliary_keys:
            for aux_key in auxiliary_keys:
                if aux_key in data and fit_on_split in data[aux_key]:
                    fit_auxiliary_data[aux_key] = data[aux_key][fit_on_split]
        
        # Fit the transform
        transform.fit_data(fit_data, auxiliary_data=fit_auxiliary_data if fit_auxiliary_data else None)
        
        # Apply to all splits
        direct_strategy = DirectFittingStrategy()
        return direct_strategy.fit_transform(transform, data, apply_to_key, auxiliary_keys)

class MultiSourceFittingStrategy(FittingStrategy):
    """Strategy for transforms that fit across multiple sources."""
    
    def __init__(self, aggregation_func: Optional[Callable[[List[Any]], Any]] = None):
        self.aggregation_func = aggregation_func or self._default_concatenate
    
    def _default_concatenate(self, data_segments: List[Any]) -> Any:
        """Default aggregation strategy."""
        import numpy as np
        return np.concatenate(data_segments, axis=0)
    
    def fit_transform(self, 
                     transform: BaseTransform,
                     data: List[Dict[str, Any]],
                     apply_to_key: str,
                     fit_on_split: str,
                     auxiliary_keys: Optional[List[str]] = None,
                     **kwargs) -> List[Dict[str, Any]]:
        
        if not isinstance(data, list):
            raise TypeError("MultiSourceFittingStrategy expects a list of data dictionaries")
        
        if fit_on_split is None:
            raise ValueError("fit_on_split must be specified")
        
        if not transform.supports_multi_source_fit():
            raise ValueError(f"Transform {transform.__class__.__name__} does not support multi-source fitting")
        
        # Collect data from all sources for fitting
        fit_data_segments = []
        fit_auxiliary_segments = {aux_key: [] for aux_key in (auxiliary_keys or [])}
        
        for source_data in data:
            fit_data_segments.append(source_data[apply_to_key][fit_on_split])
            
            # Collect auxiliary data
            if auxiliary_keys:
                for aux_key in auxiliary_keys:
                    if aux_key in source_data and fit_on_split in source_data[aux_key]:
                        fit_auxiliary_segments[aux_key].append(source_data[aux_key][fit_on_split])
        
        # Aggregate data for fitting
        aggregated_fit_data = self.aggregation_func(fit_data_segments)
        
        # Aggregate auxiliary data
        aggregated_auxiliary_data = {}
        if auxiliary_keys:
            for aux_key, aux_segments in fit_auxiliary_segments.items():
                if aux_segments:  # Only aggregate if we have data
                    aggregated_auxiliary_data[aux_key] = self.aggregation_func(aux_segments)
        
        # Fit the transform
        transform.fit_data(
            aggregated_fit_data, 
            auxiliary_data=aggregated_auxiliary_data if aggregated_auxiliary_data else None
        )
        
        # Apply to each source
        direct_strategy = DirectFittingStrategy()
        result = []
        for source_data in data:
            transformed_source = direct_strategy.fit_transform(
                transform, source_data, apply_to_key, auxiliary_keys
            )
            result.append(transformed_source)
        
        return result

# ============================================================================
# STRATEGY FACTORY
# ============================================================================

class StrategyFactory:
    """Factory to create fitting strategies based on metadata."""
    
    @staticmethod
    def create_strategy(metadata: Dict[str, Any]) -> FittingStrategy:
        """Create appropriate fitting strategy based on metadata."""
        
        fit_on = metadata.get('fit_on')
        multi_source_fit = metadata.get('multi_source_fit', False)
        
        if multi_source_fit:
            if fit_on is None:
                raise ValueError("Multi-source fitting requires 'fit_on' to be specified")
            return MultiSourceFittingStrategy()
        elif fit_on is not None:
            return SingleSourceFittingStrategy()
        else:
            return DirectFittingStrategy()

# ============================================================================
# UPDATED DATATRANSFORM CLASS
# ============================================================================

class DataTransform:
    """
    Updated DataTransform that uses composition with fitting strategies.
    Maintains the same interface and configuration-driven approach.
    """
    
    def __init__(self, transform_name: str, transform: dict, metadata: dict):
        self.transform_name = transform_name
        self.metadata = metadata
        
        # Extract metadata
        self.apply_to = metadata.get('apply_to')
        self.fit_on = metadata.get('fit_on')
        self.auxiliary_keys = metadata.get('auxiliary_keys')
        self.multi_source_fit = metadata.get('multi_source_fit', False)
        
        if self.apply_to is None:
            raise ValueError(f"Metadata for transform '{transform_name}' must specify 'apply_to' key")
        
        # Create the transform instance
        if isinstance(transform, DictConfig):
            self.transform_instance: BaseTransform = instantiate(transform)
        else:
            self.transform_instance = transform
        
        if not isinstance(self.transform_instance, BaseTransform):
            raise TypeError(
                f"Transform instance for '{transform_name}' must inherit from BaseTransform"
            )
        
        # Create the appropriate fitting strategy
        self.fitting_strategy = StrategyFactory.create_strategy(metadata)
    
    def forward(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Apply the transformation using the composed fitting strategy.
        This maintains the same interface as before.
        """
        
        # Apply the fitting strategy
        transformed_data = self.fitting_strategy.fit_transform(
            transform=self.transform_instance,
            data=data,
            apply_to_key=self.apply_to,
            fit_on_split=self.fit_on,
            auxiliary_keys=self.auxiliary_keys
        )
        
        # Validation is now handled by the strategies, but we can add
        # additional validation here if needed
        self._validate_output(data, transformed_data)
        
        return transformed_data
    
    def _validate_output(self, original_data, transformed_data):
        """Basic validation of the transformed output."""
        if isinstance(original_data, dict) and not isinstance(transformed_data, dict):
            raise TypeError(f"Transform '{self.transform_name}' changed data type unexpectedly")
        
        if isinstance(original_data, list) and not isinstance(transformed_data, list):
            raise TypeError(f"Transform '{self.transform_name}' changed data type unexpectedly")
        
        if isinstance(original_data, list) and len(original_data) != len(transformed_data):
            raise ValueError(f"Transform '{self.transform_name}' changed number of sources")

# ============================================================================
# EXAMPLE TRANSFORMS WITH SIMPLIFIED INTERFACE
# ============================================================================

class SklearnTransformAdapter(BaseTransform):
    """Adapter for sklearn transforms with simplified interface."""
    
    def __init__(self, sklearn_transform):
        self.sklearn_transform = sklearn_transform
        self._is_fitted = False
    
    def fit_data(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None) -> 'SklearnTransformAdapter':
        """Fit the sklearn transform."""
        self.sklearn_transform.fit(data)
        self._is_fitted = True
        return self
    
    def transform_data(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None, 
                      mode: Optional[str] = None) -> Any:
        """Transform data using sklearn transform."""
        if hasattr(self.sklearn_transform, 'transform'):
            return self.sklearn_transform.transform(data)
        else:
            # For transforms that don't have separate fit/transform
            return self.sklearn_transform.fit_transform(data)
    
    def supports_multi_source_fit(self) -> bool:
        """Most sklearn transforms can handle concatenated data."""
        return True

class ConcatenateTransform(BaseTransform):
    """Example transform that uses auxiliary data."""
    
    def transform_data(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None, 
                      mode: Optional[str] = None) -> Any:
        """Concatenate main data with auxiliary data."""
        if auxiliary_data is None:
            return data
        
        import numpy as np
        
        # Concatenate with auxiliary data
        all_data = [data]
        for aux_key, aux_data in auxiliary_data.items():
            all_data.append(aux_data)
        
        return np.concatenate(all_data, axis=-1)  # Concatenate along feature dimension
    
    def supports_multi_source_fit(self) -> bool:
        """Concatenation doesn't require fitting."""
        return True

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example configuration (same as your current format)
    config = {
        'scaler_features': {
            'transform': {'_target_': 'sklearn.preprocessing.StandardScaler'},
            'metadata': {
                'apply_to': 'features',
                'fit_on': 'train'
            }
        },
        'concatenate_features': {
            'transform': {'_target_': '__main__.ConcatenateTransform'},
            'metadata': {
                'apply_to': 'features',
                'auxiliary_keys': ['timestamps']
            }
        }
    }
    
    # Create transforms exactly as before
    # scaler = DataTransform('scaler_features', config['scaler_features']['transform'], config['scaler_features']['metadata'])
    # concat = DataTransform('concatenate_features', config['concatenate_features']['transform'], config['concatenate_features']['metadata'])
    
    # Usage remains the same
    # transformed_data = scaler.forward(data)
    # final_data = concat.forward(transformed_data)





class MultiSourceTransformMixin():
    """
    Stateless mixin for transforming data from multiple sources.
    """
    def __init__(self):
        """
        Initialize the MultiSourceTransformMixin.
        """

        # Check if the class is a subclass of TransformerMixin
        if not hasattr(self, 'fit_data') or not hasattr(self, 'transform_data'):
            raise TypeError("MultiSourceTransformMixin requires the class to implement fit and transform methods.")


    def fit_sources(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None, 
                      mode: Optional[str] = None) -> None:
        """
        Get all train sources and fit the scalers to them.
        """
        
        # SOME LOGIC HERE

        # Choose how to handle the multisource data
        data = np.concatenate(data, axis=0)
        self.fit_data(data)

    def transform_sources(self, data: Any, auxiliary_data: Optional[Dict[str, Any]] = None, 
                      mode: Optional[str] = None) -> None:
        """
        Transform the data from all sources using the fitted scalers.
        """

        # SOME LOGIC HERE 
        transformed_data = self.transform_data(data)
                
        return