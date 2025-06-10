from typing import Dict, List, Optional

import numpy as np
from box import Box
from omegaconf import DictConfig, OmegaConf
from sklearn.base import TransformerMixin

from lmetk.data.datasources.base import DataSourceLoader
# from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter
from hydra.utils import instantiate 
from collections import OrderedDict

class MultiSource:
    def __init__(self, **kwargs):
        self.source_names = kwargs.get('source_list', [])
        self.sources = []
        self.source_params = []
        self.transforms_list = []

        for key in self.source_names:
            # Store params as dict
            src_params = OmegaConf.to_container(kwargs[key], resolve=True)
            self.source_params.append(src_params)

            # Instantiate datasource
            datasource = instantiate(kwargs[key])
            if not getattr(datasource, '_is_loaded', False):
                raise RuntimeError(f'Source {key} has not been loaded')
            if not getattr(datasource, '_is_splitted', False):
                raise RuntimeError(f'Source {key} has not been splitted')

            self.sources.append(datasource.get_data())
            self.transforms_list.append(datasource.get_transforms())

        # Multisource transforms
        self.multisource_transforms = OrderedDict(kwargs['multisource_transforms']) if 'multisource_transforms' in kwargs else None
        self._is_loaded = True

    def __repr__(self) -> str:
        return f"MultiSourceDataSource(sources={len(self.sources)} units)"

    def get_data(self) -> list[dict]:
        """Return data sources."""
        return self.sources

    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.source_names

    def get_transforms(self) -> list[dict]:
        """Return data transforms per source."""
        return self.transforms_list

    def get_multisource_transforms(self):
        """Return multisource data transforms."""
        return self.multisource_transforms

    
