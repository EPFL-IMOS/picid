"""Preprocessor for data."""

import json
import os
import time
import hydra
import torch

import torch
from typing import Any, Dict, List, Tuple, Optional
from omegaconf import DictConfig
from collections import OrderedDict

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

    def prepare_multisource_transforms(self,):
        multisource_transforms = self.datasource.get_multisource_transforms() if hasattr(self.datasource, 'get_multisource_transforms') else None
        
        if multisource_transforms: 
            # Instantiate transforms 
            multisource_transforms = hydra.utils.instantiate(multisource_transforms)
            # OrderedDict
            # # Check that source names are in availbale sources
            # for transform_name, transform_sources in multisource_transforms.items():
            #     for transform, sources in transform_sources.items():
            #         for source in sources:
            #             if source not in 



    def pipeline(self):
        """Run the preprocessing pipeline."""
        self.prepare_data()
        self.prepare_multisource_transforms()
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
        
        # Further processing logic can use self.data_lst, self.data_name_lst, self.data_transforms_lst

    # def instantiate_pre_transform(self, transforms_config: Optional[DictConfig]):
    #     """Instantiate the pre-transforms.

    #     Parameters
    #     ----------
    #     transforms_config : DictConfig
    #         Configuration parameters for the transforms.

    #     Returns
    #     -------
    #     torch_geometric.transforms.Compose or None
    #         Pre-transform object.
    #     """
    #     if transforms_config is None:
    #         return None
    #     pre_transforms = hydra.utils.instantiate(transforms_config)
    #     return pre_transforms

    # Example for saving transform parameters (uncomment and adapt as needed)
    # def save_transform_parameters(self) -> None:
    #     """Save the transform parameters."""
    #     path_transform_parameters = os.path.join(
    #         self.processed_data_dir, "path_transform_parameters_dict.json"
    #     )
    #     if not os.path.exists(path_transform_parameters):
    #         with open(path_transform_parameters, "w") as f:
    #             json.dump(self.transforms_parameters, f, indent=4)
    #     else:
    #         with open(path_transform_parameters) as f:
    #             saved_transform_parameters = json.load(f)
    #         if saved_transform_parameters != self.transforms_parameters:
    #             raise ValueError(
    #                 "Different transform parameters for the same data_dir"
    #             )
    #         print(
    #             f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
    #         )
        

    