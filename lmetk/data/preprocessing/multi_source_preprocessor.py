import os
import time
from typing import Any, Union, Dict, List, Optional
from collections import OrderedDict
from numpy import ndarray

import hydra
from omegaconf import DictConfig, OmegaConf



from lmetk.transforms.data_transform import DataTransform
from lmetk.data.preprocessing.base import PreProcessorInterface
import numpy as np

class MultiSourcePreProcessor(PreProcessorInterface):
    """Preprocessor of data."""

    def __init__(self, datasource, transforms, logger=None,**kwargs):
        super().__init__(
            datasource=datasource, 
            transforms=transforms, 
            logger=logger
        )
        
        self.data_lst: Optional[List[Any]] = None
        self._is_preprocessed = False

    def get_processed_data(self) -> dict[str, ndarray | list[ndarray]]:
        """
        Returns the final processed data.

        The returned dictionary is structured as follows:
        {
            "features": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "target": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "other_named_tensor": {
                ...
            }
        }
        """
        if not self._is_preprocessed or self.data_lst is None:
            raise RuntimeError("Data has not been preprocessed. Please run the pipeline() first.")

        if isinstance(self.data_lst, list) and len(self.data_lst) == 1:
            return self.data_lst[0]
        
        elif isinstance(self.data_lst, list):
            # TODO: HAVE NOT TESTED
            # If multiple sources, return as a list
            return convert_outer_list_to_inner(self.data_lst)
        
        else:
            raise ValueError("Processed data is in an unexpected format.")
    
    # # TODO: that is needed at least for evaluator, extend it for multisource maybe
    # def get_transform_names(self,):
    #     return self.source_transform_names_lst

    def prepare_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        data_lst = self.datasource.get_data()

        # Ensure all outputs are lists for consistent processing
        if not isinstance(data_lst, list):
            data_lst = [data_lst]
        
        self.data_lst = data_lst

    def apply_transforms(self, data_lst: List[Any], transforms) -> List[Any]:
        """
        Apply a list of transforms to each data item in data_lst.
        Args:
            data_lst: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        if not all(isinstance(t, DataTransform) for t in transforms.values()):
            raise ValueError("Every element of transforms_lst must be a DataTransform object.")
        
        transformed_data_lst = []
        for current_data in data_lst:
            for transform_name, transform in transforms.items():
                # Removed detailed print statements for each transform application.
                if self.logger:
                    self.logger.info(f"Applying transform: '{transform_name}' (Strategy: {transform.strategy.__class__.__name__})")
                start_time = time.time()
                current_data = transform.forward(current_data)
                end_time = time.time()

                if self.logger:
                    self.logger.info(f" transform: '{transform.transform_name}' - completed in {end_time - start_time:.4f} seconds.")
            
            transformed_data_lst.append(current_data)
        return transformed_data_lst

    def pipeline(self):
        """Run the preprocessing pipeline."""
        self.prepare_data()
        
        assert self.datasource.get_split_mode() == "cross_unit", "That is a multisource preprocessing hence datasource {self.datasource.get_split_mode()} should have 'cross_unit' split mode, please define the multisource data splitter"
        self.data_lst = self.apply_transforms(self.data_lst, self.transforms.get_transforms())    
        self._is_preprocessed = True
        
        return self.data_lst

def convert_outer_list_to_inner(data_list):
    if not data_list:
        return {}

    # Ensure all dictionaries have the same keys
    keys = data_list[0].keys()
    for d in data_list:
        if d.keys() != keys:
            raise ValueError("Not all dicts have the same keys!")

    # Concatenate values for each key
    stacked = {}
    for key in keys:
        try:
            stacked[key] = [d[key] for d in data_list]
        except Exception as e:
            raise ValueError(f"Failed to concatenate key '{key}': {e}")

    # Check that all keys in the stacked have the same length
    lengths = [len(stacked[key]) for key in stacked]
    if len(set(lengths)) != 1:
        raise ValueError("All keys in the stacked dict must have the same length.")
    
    return stacked  