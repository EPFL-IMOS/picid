from collections import OrderedDict
from typing import Dict, List, Any, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from lmetk.transforms.data_transform import DataTransform

import logging
logger = logging.getLogger(__name__) 

class ConfigTransformManager:
    """
    Improved configuration-driven transform manager.
    Instantiates DataTransform objects eagerly by default and provides efficient access.
    """
    
    def __init__(self, 
                 transforms_config: Optional[Union[Dict, DictConfig]] = None,
                 lazy_instantiation: bool = False):
        """
        Initialize transform manager with configuration.
        
        Args:
            transforms_config: Transforms configuration dictionary or DictConfig
            lazy_instantiation: If False (default), instantiate transforms immediately
        """
        self.config = self._ensure_dict_config(transforms_config or {})
        self.lazy_instantiation = lazy_instantiation
        
        # Core storage - OrderedDict for efficient access to DataTransform objects
        self._data_transforms: Optional[OrderedDict[str, DataTransform]] = None
        self._is_instantiated = False
        
        # Instantiate immediately unless lazy mode
        if not lazy_instantiation and self.config:
            self._instantiate_all()
    
    def _ensure_dict_config(self, config: Union[Dict, DictConfig]) -> DictConfig:
        """Ensure config is a DictConfig."""
        if isinstance(config, dict):
            return OmegaConf.create(config)
        return config
    
    def _instantiate_all(self) -> None:
        """Instantiate all DataTransform objects from configuration."""
        if self._is_instantiated:
            return
            
        data_transforms = OrderedDict()
        
        for transform_name, transform_entry_config in self.config.items():
            # Get the transform configuration
            transform_config = transform_entry_config.get('transform')
            
            if transform_config is None:
                logger.error(f"Transform '{transform_name}' missing 'transform' configuration")
            
            # Instantiate the raw transform using Hydra
            instantiated_transform = instantiate(transform_config)

            # assert the instantiated_transform is instantiated:
            if instantiated_transform is None:
                logger.error(f"Failed to instantiate transform '{transform_name}' from config: {transform_config}")
                raise RuntimeError(f"Failed to instantiate transform '{transform_name}' from config: {transform_config}")

            
            # Get metadata
            metadata = transform_entry_config.get('metadata', {})
            
            # Create DataTransform object
            data_transform = DataTransform(
                transform_name=transform_name,
                transform=instantiated_transform,
                metadata=metadata
            )
            
            data_transforms[transform_name] = data_transform
            


            
        
        self._data_transforms = data_transforms
        self._is_instantiated = True
    
    def add_transforms_config(self, transforms_config: Union[Dict, DictConfig]) -> None:
        """
        Add transforms from configuration.
        
        Args:
            transforms_config: Configuration dictionary with transform definitions
            
        Raises:
            RuntimeError: If transforms are already instantiated (must clear cache first)
            KeyError: If transform name already exists
        """
        if self._is_instantiated:
            raise RuntimeError(
                "Cannot add transforms after instantiation. "
                "Call clear_cache() first to reset the manager state."
            )
        
        new_config = self._ensure_dict_config(transforms_config)
        
        # Check for conflicts
        conflicts = [name for name in new_config.keys() if name in self.config]
        if conflicts:
            raise KeyError(f"Transform names already exist: {conflicts}")
        
        # Merge with existing config
        self.config = OmegaConf.merge(self.config, new_config)
        
        # Auto-instantiate if not in lazy mode
        if not self.lazy_instantiation:
            self._instantiate_all()
    
    def update_transforms_config(self, transforms_config: Union[Dict, DictConfig]) -> None:
        """
        Update/replace transforms configuration.
        
        Args:
            transforms_config: New transforms configuration
            
        Raises:
            RuntimeError: If transforms are already instantiated (must clear cache first)
        """
        if self._is_instantiated:
            raise RuntimeError(
                "Cannot update transforms after instantiation. "
                "Call clear_cache() first to reset the manager state."
            )
        
        self.config = self._ensure_dict_config(transforms_config)
        
        # Auto-instantiate if not in lazy mode
        if not self.lazy_instantiation and self.config:
            self._instantiate_all()
    
    def has_transform(self, name: str) -> bool:
        """Check if transform name exists in configuration."""
        return name in self.config
    
    def get_transform_names(self) -> List[str]:
        """Get list of all transform names in configuration order."""
        return list(self.config.keys())
    
    def get_transforms_config(self) -> DictConfig:
        """Get the transforms configuration."""
        return self.config
    
    def get_transform_config(self, name: str) -> DictConfig:
        """
        Get configuration for a specific transform.
        
        Args:
            name: Transform name
            
        Returns:
            Transform configuration
        """
        if name not in self.config:
            raise KeyError(f"Transform '{name}' not found")
        
        return self.config[name]
    
    def get_data_transforms(self) -> OrderedDict[str, DataTransform]:
        """
        Get all instantiated DataTransform objects as OrderedDict.
        This is the primary method for accessing DataTransform objects.
        
        Returns:
            OrderedDict with all instantiated DataTransform objects
        """
        if not self._is_instantiated:
            self._instantiate_all()
        
        return self._data_transforms.copy() if self._data_transforms else OrderedDict()
    
    def get_data_transform(self, name: str) -> DataTransform:
        """
        Get specific instantiated DataTransform object by name.
        Efficient - no re-instantiation, direct access from memory.
        
        Args:
            name: Transform name
            
        Returns:
            DataTransform object
        """
        if not self._is_instantiated:
            self._instantiate_all()
        
        if not self._data_transforms or name not in self._data_transforms:
            raise KeyError(f"Transform '{name}' not found")
        
        return self._data_transforms[name]
    
    def prepare_transforms(self) -> List[DataTransform]:
        """
        Get DataTransform objects as list.
        Efficient - returns from already instantiated DataTransform objects.
        
        Returns:
            List of DataTransform objects
        """
        data_transforms = self.get_data_transforms()
        return list(data_transforms.values())
    
    def get_transforms_by_apply_to(self, apply_to: str) -> List[DataTransform]:
        """
        Get DataTransform objects that apply to a specific data type.
        
        Args:
            apply_to: Data type (e.g., 'features', 'target', 'timestamps')
            
        Returns:
            List of DataTransform objects filtered by apply_to
        """
        data_transforms = self.get_data_transforms()
        return OrderedDict(
            (dt.transform_name, dt)
            for dt in data_transforms.values()
            if dt.metadata.get('apply_to') == apply_to
        )
    
    def get_transforms_by_fit_on(self, fit_on: str) -> List[DataTransform]:
        """
        Get DataTransform objects that should be fit on specific split.
        
        Args:
            fit_on: Split name (e.g., 'train', 'val')
            
        Returns:
            List of DataTransform objects filtered by fit_on
        """
        data_transforms = self.get_data_transforms()
        return OrderedDict(
            (dt.transform_name, dt)
            for dt in data_transforms.values()
            if dt.metadata.get('fit_on') == fit_on
        )
    
    def get_transforms_with_auxiliary_keys(self) -> List[DataTransform]:
        """
        Get DataTransform objects that have auxiliary_keys in their metadata.
        
        Returns:
            List of DataTransform objects with auxiliary keys
        """
        data_transforms = self.get_data_transforms()
        return OrderedDict(
            (dt.transform_name, dt) 
            for dt in data_transforms.values() 
            if 'auxiliary_keys' in dt.metadata
        )
    
    def remove_transform(self, name: str) -> None:
        """
        Remove a transform from configuration.
        
        Args:
            name: Transform name to remove
            
        Raises:
            RuntimeError: If transforms are already instantiated (must clear cache first)
        """
        if self._is_instantiated:
            raise RuntimeError(
                "Cannot remove transforms after instantiation. "
                "Call clear_cache() first to reset the manager state."
            )
        
        if name not in self.config:
            raise KeyError(f"Transform '{name}' not found")
        
        del self.config[name]
    
    def clear_cache(self) -> None:
        """
        Clear instantiated DataTransform objects and reset manager state.
        After calling this, you can modify configuration again.
        """
        self._data_transforms = None
        self._is_instantiated = False
    
    def force_reinstantiate(self) -> None:
        """
        Force re-instantiation of all transforms from current configuration.
        Useful when you want to reload transforms without clearing config.
        """
        self.clear_cache()
        if self.config:
            self._instantiate_all()
    
    @property
    def is_instantiated(self) -> bool:
        """Check if transforms have been instantiated."""
        return self._is_instantiated
    
    # Backward compatibility methods
    def instantiate_transforms(self) -> OrderedDict[str, DataTransform]:
        """Alias for get_data_transforms() for backward compatibility."""
        return self.get_data_transforms()
    
    def get_transforms(self) -> OrderedDict[str, DataTransform]:
        """Alias for get_data_transforms() for backward compatibility."""
        return self.get_data_transforms()
    
    def get_transform(self, name: str) -> DataTransform:
        """Alias for get_data_transform() for backward compatibility."""
        return self.get_data_transform(name)
    
    def __len__(self) -> int:
        """Return total number of transforms."""
        return len(self.config)
    
    def __contains__(self, name: str) -> bool:
        """Check if transform exists using 'in' operator."""
        return self.has_transform(name)
    
    def __repr__(self) -> str:
        """String representation."""
        status = "instantiated" if self._is_instantiated else "not instantiated"
        return f"ConfigTransformManager(transforms={len(self.config)}, {status})"


# Factory function for easy creation
def create_transform_manager_from_config(config_path: Optional[str] = None,
                                       config_dict: Optional[Dict] = None,
                                       lazy_instantiation: bool = False) -> ConfigTransformManager:
    """
    Create transform manager from configuration file or dictionary.
    
    Args:
        config_path: Path to YAML configuration file
        config_dict: Configuration dictionary
        lazy_instantiation: If False (default), instantiate transforms immediately
        
    Returns:
        ConfigTransformManager instance
    """
    if config_path:
        config = OmegaConf.load(config_path)
    elif config_dict:
        config = OmegaConf.create(config_dict)
    else:
        config = OmegaConf.create({})
    
    return ConfigTransformManager(
        transforms_config=config, 
        lazy_instantiation=lazy_instantiation
    )