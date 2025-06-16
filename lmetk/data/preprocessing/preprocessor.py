import os
import time
from typing import Any, Union, Dict, List, Optional
from collections import OrderedDict



import hydra
from omegaconf import DictConfig, OmegaConf



from lmetk.transforms.data_transform import DataTransform
import numpy as np


class PreProcessor:
    """Preprocessor of data."""

    def __init__(self, datasource, **kwargs):
        self.datasource = datasource
        
        self.data_lst: Optional[List[Any]] = None
        self.data_name_lst: Optional[List[Any]] = None
        self.source_transforms_lst: Optional[List[Any]] = None
        self.source_transform_names_lst: Optional[List[Any]] = None
        self.multisource_transforms: Optional[List[Any]] = None
        self.multisource_data_splitter: Optional[Any] = None
        self._is_preprocessed = False

    def get_processed_data(self,):
        return self.data_lst, self.source_transforms_lst, self.multisource_transforms
    
    # TODO: that is needed at least for evaluator, extend it for multisource maybe
    def get_transform_names(self,):
        return self.source_transform_names_lst

    def prepare_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        data_lst = self.datasource.get_data()
        data_name_lst = self.datasource.get_data_name()
        source_transforms_raw = self.datasource.get_transforms() 
        source_transform_names_raw = self.datasource.get_transform_names()
        multisource_data_splitter = self.datasource.get_multisource_data_splitter()

        # Ensure all outputs are lists for consistent processing
        if not isinstance(data_lst, list):
            data_lst = [data_lst]
        
        if not isinstance(data_name_lst, list):
            data_name_lst = [data_name_lst]
        
        # Ensure source_transforms_raw is a list, as MultiSource.get_transforms() returns a list
        # and DataSourceLoader.get_transforms() returns a single OrderedDict.
        if not isinstance(source_transforms_raw, list):
            source_transforms_raw = [source_transforms_raw]

        self.data_lst = data_lst
        self.data_name_lst = data_name_lst
        self.source_transforms_lst = source_transforms_raw
        self.source_transform_names_lst = source_transform_names_raw
        self.multisource_data_splitter = multisource_data_splitter


    def prepare_multisource_transforms(self) -> None:
        """Prepare and store multisource transforms from the datasource."""
        multisource_transforms_config = None
        if hasattr(self.datasource, 'get_multisource_transforms'):
            
            config_from_method = self.datasource.get_multisource_transforms()
            if isinstance(config_from_method, (OrderedDict, DictConfig)):
                multisource_transforms_config = config_from_method
            # Removed the warning for unexpected types, as it's a demonstration print.
            # elif config_from_method is not None:
            #     print(f"Warning: get_multisource_transforms() returned unexpected type: {type(config_from_method)}. Expected OrderedDict or DictConfig. Skipping multi-source transforms.")
        # Removed the print for no method found.
        # else:
        #     print("Datasource has no 'get_multisource_transforms' method. No multi-source transforms will be applied.")

        self.multisource_transforms = [self._prepare_transforms(multisource_transforms_config)]


    def prepare_source_transforms(self) -> None:
        """
        Prepare and update source transforms list by instantiating them.
        This method assumes `self.source_transforms_lst` has already been populated
        with the raw transform configurations (OrderedDicts or DictConfigs) by `prepare_data()`.
        """
        if self.source_transforms_lst is None:
            raise ValueError("Source transforms not initialized. Call prepare_data() first.")
        
        if len(self.data_lst) != len(self.source_transforms_lst):
            raise ValueError(
                f"Mismatch: Number of loaded data sources ({len(self.data_lst)}) "
                f"does not match number of source transform configurations ({len(self.source_transforms_lst)})."
            )

        instantiated_source_transforms = []
        for transforms_config_for_source in self.source_transforms_lst:
            instantiated_source_transforms.append(
                self._prepare_transforms(transforms_config_for_source)
            )
        self.source_transforms_lst = instantiated_source_transforms


    def _prepare_transforms(self, transforms_config: Optional[Union[DictConfig, OrderedDict]]) -> List[DataTransform]:
        """
        Convert a dictionary/DictConfig of transforms into a list of DataTransform objects.
        This method is responsible for using Hydra's instantiate on the 'transform' part.
        """
        if not transforms_config:
            return []
        
        out_transforms: List[DataTransform] = []
        if not isinstance(transforms_config, OrderedDict):
            raise ValueError('transforms_config should be OrderedDict at this stage of pipeline')

        items_to_process = transforms_config.items() 

        for transform_name, transform_entry_config in items_to_process:
            
            instantiated_transform_obj = transform_entry_config.get('transform')
            metadata = transform_entry_config.get('metadata', {})
            out_transforms.append(
                DataTransform(
                    transform_name=transform_name,
                    transform=instantiated_transform_obj,
                    metadata=metadata
                )
            )

        return out_transforms
        

    def _apply_transforms(self, data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]], transforms: List[DataTransform]) -> Any:
        """Apply all transforms in sequence to the data."""
        current_data = data
        for transform in transforms:
            # Removed detailed print statements for each transform application.
            # print(f"  Applying transform: '{transform.transform_name}' (Strategy: {transform.strategy.__class__.__name__})")
            # start_time = time.time()
            current_data = transform.forward(current_data)
            # end_time = time.time()
            # print(f"    - Completed in {end_time - start_time:.4f} seconds.")
            
            # Removed detailed logging of data after each transform.
        return current_data

    def apply_source_transforms(self, data_lst: List[Any], transforms_lst: List[List[DataTransform]]) -> List[Any]:
        """
        Apply a list of transforms to each data item in data_lst.
        Args:
            data_lst: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        # Removed "--- Applying Source-Specific Transforms ---" print.
        if not isinstance(transforms_lst, list) or not all(isinstance(t, list) for t in transforms_lst):
            raise ValueError("transforms_lst must be a list of lists of DataTransform objects.")
        if len(data_lst) != len(transforms_lst):
            raise ValueError(f"Length mismatch: data_lst ({len(data_lst)}) and transforms_lst ({len(transforms_lst)}) must have the same length.")

        transformed_data_lst = []
        for data, transforms in zip(data_lst, transforms_lst):
            # Removed per-source processing print.
            # print(f"\nProcessing source {i+1}/{len(data_lst)}: {self.data_name_lst[i] if self.data_name_lst else f'Source_{i}'}")
            transformed_data_lst.append(self._apply_transforms(data, transforms))
        return transformed_data_lst

    def apply_multisource_transforms(self, data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]], transforms: List[DataTransform]) -> Any:
        """
        Apply multisource transforms to the data.
        Args:
            data: The data to transform.
            transforms: List of DataTransform objects.
        Returns:
            Transformed data.
        """
        if not transforms:
            # Removed "No multi-source transforms to apply." print.
            return data

        # Removed "--- Applying Multi-Source Transforms ---" print.
        return self._apply_transforms(data, transforms)


    def pipeline(self):
        """Run the preprocessing pipeline."""
        # Removed "Starting preprocessing pipeline..." print.
        self.prepare_data()
        
        # if multisource_data_splitter do not exist, then we work with separate sources
        if not self.multisource_data_splitter:
            self.prepare_source_transforms()
            self.data_lst = self.apply_source_transforms(self.data_lst, self.source_transforms_lst)
        else:
            # First reasemble sources: self.data_lst
            self.prepare_multisource_transforms()
            self.data_lst = self.apply_source_transforms(self.data_lst, self.multisource_transforms) #self.apply_multisource_transforms(self.data_lst, self.multisource_transforms)
        
        self._is_preprocessed = True
        
        return self.data_lst

def key_exists_anywhere(data, target_key):
    # Check if object has dict-like methods
    if hasattr(data, 'keys') and hasattr(data, 'values') and hasattr(data, 'items'):
        # First check if the target key exists at this level
        if target_key in data:
            return True, data[target_key]
        
        # Then recursively search in all values
        for value in data.values():
            found, val = key_exists_anywhere(value, target_key)
            if found:
                return found, val
                
    elif isinstance(data, (list, tuple)):
        for item in data:
            found, val = key_exists_anywhere(item, target_key)
            if found:
                return found, val
    
    return False, None
        