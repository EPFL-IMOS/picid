import os
import time
from typing import Any, Union, Dict, List, Optional
from collections import OrderedDict
from numpy import ndarray

import hydra
from omegaconf import DictConfig, OmegaConf



from lmetk.transforms.data_transform import DataTransform
import numpy as np
from abc import ABC, abstractmethod
from lmetk.data.datasources import DataSourceLoader
from lmetk.transforms.transform_manager import ConfigTransformManager


class PreProcessorInterface(ABC):
    """Preprocessor of data."""

    def __init__(self, 
                 datasource: Union[DataSourceLoader, DataSourceLoader], 
                 transforms: ConfigTransformManager,
                 logger=None,
                 **kwargs):
        self.datasource = datasource
        self.transforms = transforms
        self.logger = logger
        
        

    @abstractmethod
    def get_processed_data(self,) ->  dict[str, dict[str, ndarray | list[ndarray]]]:
        """
        Returns the final processed data
        the dictionary should be structured as follows:
        - features:
            - train: ndarray | list[ndarray]
            - val: ndarray | list[ndarray]
            - test: ndarray | list[ndarray]
        - target:
            - train: ndarray | list[ndarray]
            - val: ndarray | list[ndarray]
            - test: ndarray | list[ndarray]
        - other_named_tensor:
            - ...
        """
        raise NotImplementedError("get_processed_data method not implemented.") 
    
    @abstractmethod
    def prepare_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        raise NotImplementedError("prepare_data method not implemented.")

    # @abstractmethod
    # def prepare_transforms(self) -> None:
    #     """Prepare and store transforms required to be applied to the datasource."""
    #     raise NotImplementedError("prepare_transforms method not implemented.")

    @abstractmethod
    def apply_transforms(self) -> None:
        """Apply the transforms to the datasource."""
        raise NotImplementedError("apply_transforms method not implemented.")

    @abstractmethod
    def pipeline(self):
        """Run the preprocessing pipeline."""
        raise NotImplementedError("pipeline method not implemented.")


   