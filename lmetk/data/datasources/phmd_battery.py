from typing import Dict
from hydra.utils import instantiate 
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from lmetk.data.datasources.base import DataSourceLoaderInterface
from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter
import numpy as np
from phmd import datasets
import logging
logger = logging.getLogger(__name__)

class PHMDMultiSourceLoader(DataSourceLoaderInterface):
    """This class instantiates each of the soure"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.data_source_dict = {}
        #self.source_params = []    
        # Get source names as list
        #self.source_names = list(kwargs.get('source_list', {}).keys())
        self.fold = kwargs['fold']
        # Instantiate multisource_data_splitter if is not none
        assert kwargs.get('multisource_data_splitter') == None, 'For PHMD, multisource_data_splitter is not supported. PHMD are given with splits.'
        self.multisource_data_splitter = 'For PHMD, multisource_data_splitter is not supported. PHMD are given with splits.'
        
        
    def __repr__(self) -> str:
        return f"MultiSource(sources={len(self.data_source_dict)} units)"

    def get_data(self) -> dict | list[dict]:
        """Return data sources.
        
        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """

        # Load data from each source
        # Load sources
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_isplitted = True
        
        return self.data_dict
        
    def get_data_name(self) -> list[str]:
        """Return source data names."""
        return self.data_dict['metadata']
    
    def get_multisource_data_splitter(self):
        """Return multisource data splitter."""
        return self.multisource_data_splitter

    def split_data(self, data_lst: list[Dict], source_names: list[str]):
        "Predefined splits"
        pass
    
    def _load_data(self, ):
        # Load dataset and metadata
        ds = datasets.Dataset(self.data_name, cache_dir='data')
        task = ds['rul']
        folds = task[self.fold]  # fold_num = 0 by default

        unit_id = ds['rul'].meta['identifier']
        assert len(unit_id) == 1, "There should be only one identifier"
        unit_col = unit_id[0]
        features_col = ds['rul'].meta['features']
        target_col = 'rul'

        # Process each split
        data = defaultdict(list)
        for split in ['train', 'val', 'test']:
            df = folds[split]
            for unit_name in np.unique(df[unit_col]):
                df_unit = df[df[unit_col] == unit_name]
                data[split].append(process_unit(df_unit=df_unit, unit_name=unit_name, features_col=features_col, target_col=target_col))

        # Convert to desired format
        train = convert_outer_list_to_inner(data['train'])
        val = convert_outer_list_to_inner(data['val'])
        test = convert_outer_list_to_inner(data['test'])

        out_dict = {}
        for key in train.keys():
            out_dict[key] = {
                "train": train[key],
                "val": val[key],
                "test": test[key],
                }
        logger.info(f"Number of train units: {len(out_dict['features']['train'])}")
        logger.info(f"Number of val units: {len(out_dict['features']['val'])}")
        logger.info(f"Number of test units: {len(out_dict['features']['test'])}")
        return out_dict

    

    
# Helper function to process a single unit
def process_unit(df_unit, unit_name, features_col, target_col):
    df_features = df_unit[features_col].copy().reset_index(drop=True)
    df_target = df_unit[target_col].copy().reset_index(drop=True)

    if df_unit.isna().any().any():
        logger.info(f"NaN values detected in unit {unit_name}. Details:\n{df_unit.isna()}")
        df_features = df_features.dropna()  # Optionally handle df_target too

    return {
        "features": df_features,
        "timestamps": np.array(df_features.index),
        "target": df_target,
        "columns": df_features.columns,
        "target_name": target_col,
        "metadata": {
            'unit_id': unit_name,
            'unit_length': len(df_features)
        }
    }

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
        stacked[key] = [d[key] for d in data_list]
        
    # Check that all keys in the stacked have the same length
    lengths = [len(stacked[key]) for key in stacked]
    if len(set(lengths)) != 1:
        raise ValueError("All keys in the stacked dict must have the same length.")
    
    return stacked  