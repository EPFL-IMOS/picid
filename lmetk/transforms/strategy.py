from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class TransformStrategy(ABC):
    """Abstract base class for different transform strategies."""
    
    @abstractmethod
    def apply(self, transform_instance, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform strategy to the data."""
        pass


class FitTransformStrategy(TransformStrategy):
    """Strategy for transforms that need to be fitted on one split and applied to all."""
    
    def __init__(self, fit_on: str = "train"):
        self.fit_on = fit_on
    
    def apply(self, transform_instance, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit on specified split, then transform all splits."""
        # Fit on the specified split
        if self.fit_on in data:
            transform_instance.fit(data[self.fit_on])
        else:
            raise KeyError(f"Split '{self.fit_on}' not found in data. Available splits: {list(data.keys())}")
        
        # Transform all splits
        transformed_data = {}
        for split_name, split_data in data.items():
            transformed_data[split_name] = transform_instance.transform(split_data)
        
        return transformed_data


class DirectTransformStrategy(TransformStrategy):
    """Strategy for transforms that are applied directly to each split without fitting."""
    
    def apply(self, transform_instance, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform directly to all splits."""
        # Transform each split independently
        transformed_data = {}
        for split_name, split_data in data.items():
            transformed_data[split_name] = transform_instance.transform(split_data)
        
        return transformed_data