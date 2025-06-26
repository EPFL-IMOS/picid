from typing import Dict, List, Optional

import numpy as np
from box import Box
from omegaconf import DictConfig, OmegaConf

from lmetk.data.datasources.base import MultiSourceInterface
# from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter
from hydra.utils import instantiate 
from collections import OrderedDict
from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter

class MultiSource(MultiSourceInterface):
    """This class instantiates each of the soure"""
    def __init__(self, **kwargs):
        self.data_source_dict = {}
        self.source_params = []
        self.transforms_list = []
        self.multisource_data_splitter = None
        
        # Assign keys needed for the pipeline 
        self.data_name = kwargs['data_name']
        self.task_mode = kwargs['task_mode']
        
        # Get source names as list
        self.source_names = list(kwargs.get('source_list', {}).keys())
        
        # Instantiate multisource_data_splitter if is not none
        self.multisource_data_splitter = instantiate(kwargs['multisource_data_splitter']) if kwargs.get('multisource_data_splitter', None) is not None else None

        # Load sources
        for key in self.source_names:

            # Store params as dict
            src_params = OmegaConf.to_container(kwargs[key], resolve=True)
            self.source_params.append(src_params)

             # Instantiate datasource
            datasource = instantiate(kwargs[key], multisource_data_splitter=True if self.multisource_data_splitter else False)
            self.data_source_dict[key] = datasource
        
    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    def get_data(self) -> list[dict]:
        """Return data sources."""

        # Load data from each source

        data_lst = []
        for key in self.source_names:
            datasource = self.data_source_dict[key]
            if not getattr(datasource, '_is_loaded', False):
                raise RuntimeError(f'Source {key} has not been loaded')
            
            # Validate splitting logic based on presence of multisource_data_splitter
            if self.multisource_data_splitter:
                if getattr(datasource, '_is_splitted', False):
                    raise RuntimeError(
                        f"multisource_data_splitter is provided, but source '{key}' is already split. This configuration is not supported."
                    )
            else:
                if not getattr(datasource, '_is_splitted', False):
                    raise RuntimeError(
                        f"Source '{key}' must be split when multisource_data_splitter is not provided."
                    )

            data_lst.append(datasource.get_data())
            self.transforms_list.append(datasource.get_transforms())
        
        # Split data in case of multisouce data splitting
        if self.multisource_data_splitter:
            self.split_data(data_lst, self.source_names)

        # Multisource transforms
        self.multisource_transforms = OrderedDict(self._init_kwargs['multisource_transforms']) if 'multisource_transforms' in self._init_kwargs else None
        self._is_loaded = True
        
        return data_lst

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.source_names

    def get_transforms(self) -> list[dict]:
        """Return data transforms per source."""
        return self.transforms_list
    
    def get_transform_names(self) -> list[str]:
        """Return transform names."""
        return [name for name in self.multisource_transforms.keys()] if self.multisource_transforms else []

    def get_multisource_transforms(self):
        """Return multisource data transforms."""
        return self.multisource_transforms
    
    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    def split_data(self, data_lst: list[Dict], source_names: list[str]):
        assert isinstance(self.multisource_data_splitter, BySourceSplitter), "multisource_data_splitter must be an instance of BySourceSplitter"
        self.data_lst, self.source_names = self.multisource_data_splitter(data_lst, source_names)
    
    def _combine_sources(self,):
        """
        Rearage self.data_lst into one data object [dict].
        Case when we have multiple sources and multisource_data_splitter
        We have to rearenge as assume loading sources, where the source
        represents a split. 
        """


    
