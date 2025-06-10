from hydra.utils import instantiate
from lmetk.transforms.strategy import FitTransformStrategy, DirectTransformStrategy
from typing import Dict, Any
from omegaconf.dictconfig import DictConfig

class DataTransform:
    """Main transform class that coordinates different transform strategies."""
    
    def __init__(self, transform_name: str, transform: dict, metadata: dict):
        """
        Initialize DataTransform.
        """
        self.transform_name = transform_name
        self.metadata = metadata
        
        # Infer the strategy based on metadata
        fit_on = metadata.get('fit_on')
        if fit_on is not None:
            self.strategy = FitTransformStrategy(fit_on=fit_on)
        else:
            self.strategy = DirectTransformStrategy()
        
        # Initialize the actual transform instance
        # Check if already instantiated if not instantiate 
        if isinstance(transform, DictConfig):
            self.transform_instance = instantiate(transform)
        else:
            self.transform_instance = transform
            
    
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transform to the data.
        
        Parameters
        ----------
        data : dict
            Input data dictionary
            
        Returns
        -------
        dict
            Transformed data dictionary
        """
        # Make a copy to avoid modifying original data
        data_key = self.metadata.get('apply_to', None)
        
        if not data_key:
            raise KeyError(f'The apply_to is necessary parameter, please specify the key to apply the transformation, available keys {list(data.keys())}')
        
        if data_key not in data:
            raise KeyError(f"Data key '{data_key}' not found in data. Available keys: {list(data.keys())}")
        
        # Apply the strategy
        transformed_data = self.strategy.apply(
            self.transform_instance, 
            data[data_key]
        )
        
        # Ensure the data[data_key] can be updated with the output of transformed_data, 
        # the idea to overwrite the initial data, hence keys has to match perfectly 
        if not isinstance(transformed_data, dict):
            raise TypeError(f"Strategy returned {type(transformed_data)}, expected dict")
        
        # Validate that all keys match
        original_keys = set(data[data_key].keys())
        transformed_keys = set(transformed_data.keys())
        if original_keys != transformed_keys:
            raise KeyError(f"Key mismatch: original keys {original_keys} != transformed keys {transformed_keys}")
        
        # TODO: Add some other checks that we awnt to satisfy! 

        data[data_key].update(transformed_data)

        return data
    