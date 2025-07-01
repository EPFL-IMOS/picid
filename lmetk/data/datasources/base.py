from abc import ABC, abstractmethod
from typing import Dict
from box import Box
import numpy as np
from omegaconf import DictConfig

class DataSourceLoaderInterface(ABC):
    def __init__(self, **kwargs):
        # Store kwargs for repr
        self._init_kwargs = kwargs
        
        # Assign keys needed for the pipeline 
        self.data_name = kwargs['data_name']
        self.task_mode = kwargs['task_mode']
        
    def get_split_mode(self):
        multisource_data_splitter = self._init_kwargs.get('multisource_data_splitter', None)
        
        # Case 1: If multisource_data_splitter is None, split_mode is by unit
        if not multisource_data_splitter:
            split_mode = 'per_unit'
        
        else:
            split_mode = 'cross_unit'
            
        return split_mode
    
    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self._init_kwargs.items())
        return f"{self.__class__.__name__}({args})"
        
    @abstractmethod
    def get_data(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def get_data_name(self) -> str:
        pass

    @abstractmethod
    def split_data(self, kwargs: dict) -> None:
        pass