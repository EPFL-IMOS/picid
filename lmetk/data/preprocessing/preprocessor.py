"""Preprocessor for data."""

import json
import os
import time
import hydra
import torch

import torch

from typing import Any, Union,  Dict, List, Tuple, Optional
from omegaconf import DictConfig
from collections import OrderedDict

from lmetk.transforms.data_transform import DataTransform

class PreProcessor:
    """Preprocessor of data."""

    def __init__(self, datasource, **kwargs):
        self.datasource = datasource
        
        # self.transforms = self.instantiate_pre_transform(transforms_config)
        self.data_lst: Optional[List[Any]] = None
        self.data_name_lst: Optional[List[Any]] = None
        self.source_transforms_lst: Optional[List[Any]] = None
        self.multisource_transforms: Optional[List[Any]] = None


    def prepare_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        data_lst = self.datasource.get_data()
        data_name_lst = self.datasource.get_data_name()
        source_transforms_lst  = self.datasource.get_transforms()

        # Ensure all outputs are lists
        if not isinstance(data_lst, list):
            data_lst = [data_lst]
        
        if not isinstance(data_name_lst, list):
            data_name_lst = [data_name_lst]
        
        
        if not isinstance(source_transforms_lst, list):
            source_transforms_lst = [source_transforms_lst]

        self.data_lst = data_lst
        self.data_name_lst = data_name_lst
        self.source_transforms_lst = source_transforms_lst

    def prepare_multisource_transforms(self) -> None:
        """Prepare and store multisource transforms from the datasource."""
        multisource_transforms = None
        if hasattr(self.datasource, 'get_multisource_transforms'):
            multisource_transforms = self.datasource.get_multisource_transforms()
        self.multisource_transforms = self._prepare_transforms(multisource_transforms)

    def prepare_source_transforms(self) -> None:
        """Prepare and update source transforms list."""
        if self.source_transforms_lst is None:
            raise ValueError("Source transforms not initialized. Call prepare_data() first.")
        self.source_transforms_lst = [
            self._prepare_transforms(transforms) for transforms in self.source_transforms_lst
        ]

    def _prepare_transforms(self, transforms: Optional[Dict[str, Dict[str, Any]]]) -> List[DataTransform]:
        """Convert a dictionary of transforms into a list of DataTransform objects."""
        if not transforms:
            return []
        out_transforms: List[DataTransform] = []
        for transform_name, transform_metadata in transforms.items():
            transform = transform_metadata.get('transform')
            metadata = transform_metadata.get('metadata', {})
            out_transforms.append(
                DataTransform(
                    transform_name=transform_name,
                    transform=transform,
                    metadata=metadata
                )
            )
        return out_transforms
        
    def _apply_transforms(self, data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]], transforms: List[DataTransform]) -> Any:
        """Apply all transforms in sequence to the data."""
        for transform in transforms:
            data = transform.forward(data)
        return data

    def apply_source_transforms(self, data_lst: List[Any], transforms_lst: List[List[DataTransform]]) -> List[Any]:
        """
        Apply a list of transforms to each data item in data_lst.
        Args:
            data_lst: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        if not isinstance(transforms_lst, list) or not all(isinstance(t, list) for t in transforms_lst):
            raise ValueError("transforms_lst must be a list of lists of DataTransform objects.")
        if len(data_lst) != len(transforms_lst):
            raise ValueError("data_lst and transforms_lst must have the same length.")

        return [
            self._apply_transforms(data, transforms)
            for data, transforms in zip(data_lst, transforms_lst)
        ]

    def apply_multisource_transforms(self, data: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]], transforms: List[DataTransform]) -> Any:
        """
        Apply multisource transforms to the data.
        Args:
            data: The data to transform.
            transforms: List of DataTransform objects.
        Returns:
            Transformed data.
        """
        return self._apply_transforms(data, transforms)


    def pipeline(self):
        """Run the preprocessing pipeline."""
        self.prepare_data()
        self.prepare_source_transforms()
        self.prepare_multisource_transforms()
        
        self.data_lst = self.apply_source_transforms(self.data_lst, self.source_transforms_lst)
        self.data_lst = self.apply_multisource_transforms(self.data_lst, self.multisource_transforms)
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
        