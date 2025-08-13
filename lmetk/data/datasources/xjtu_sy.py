from typing import Dict
from hydra.utils import instantiate 
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from lmetk.data.datasources.base import DataSourceLoaderInterface
from lmetk.data.preprocessing.splits.by_source_splitter import BySourceSplitter
import numpy as np
from phmd import datasets
import logging
import pandas as pd
from pathlib import Path
logger = logging.getLogger(__name__)

class PHMDMultiSourceLoader(DataSourceLoaderInterface):
    """This class instantiates each of the soure"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fold = kwargs['fold']
        self.task_mode = kwargs['task_mode']
        self.cache_dir = kwargs["cache_dir"]

        cache_path = Path(self.cache_dir).expanduser()
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache_path)

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
        ds = datasets.Dataset(self.data_name, cache_dir=self.cache_dir)
        task = ds[self.task_mode]
        folds = task[self.fold]  # fold_num = 0 by default

        unit_id = ds[self.task_mode].meta['identifier']
        assert len(unit_id) == 1, "There should be only one identifier"
        unit_col = unit_id[0]

        # allign UNIBO dataset with the NB1 and NB14
        if unit_col == 'test_name':
            folds['train'] = folds['train'].rename(columns={'test_name':'unit'})
            folds['val'] = folds['val'].rename(columns={'test_name':'unit'})
            folds['test'] = folds['test'].rename(columns={'test_name':'unit'})
            unit_col = 'unit'

        features_col = ds[self.task_mode].meta['features']
        target_col = self.task_mode

        if self.data_name == 'NB1':
            # The features_col in 'NB1' is quite weird, so allign to UNIBO and NB14
            folds['train'] = folds['train'].rename(columns={'voltage_battery':'voltage', 'current_battery':'current', 'temp_battery':'temperature'})
            folds['val'] = folds['val'].rename(columns={'voltage_battery':'voltage', 'current_battery':'current', 'temp_battery':'temperature'})
            folds['test'] = folds['test'].rename(columns={'voltage_battery':'voltage', 'current_battery':'current', 'temp_battery':'temperature'})
            features_col = ['mode','voltage', 'current', 'temperature']

        
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
    df_features = df_unit.reset_index(drop=True) #df_unit[features_col].copy().reset_index(drop=True)
    df_target = df_unit[target_col].copy().reset_index(drop=True)
    #df_features = df_unit.copy().reset_index(drop=True)

    if df_unit.isna().any().any():
        logger.info(f"NaN values detected in unit {unit_name}. Details:\n{df_unit.isna()}")
        df_features = df_features.dropna()  # Optionally handle df_target too

    # check if df_target has infinity values
    if np.isinf(df_target).any():
        logger.info(f"Infinity values detected in unit {unit_name}. Details:\n{df_target}")
        # Replace inf with nan
        df_target = df_target.replace([np.inf, -np.inf], np.nan)
        # Find nan indexes
        nan_indexes = df_target[df_target.isna()].index
        # Drop nan rows from both target and features
        df_target = df_target.drop(nan_indexes)
        df_features = df_features.drop(nan_indexes)
        # Ensure the number of rows is the same
        assert len(df_target) == len(df_features), "Mismatch after dropping nan/infinity rows"

        
    return {
        "features": df_features,
        "timestamps": np.array(df_features.index),
        "target": df_target,
        "metadata": {
            'unit_id': unit_name,
            'unit_length': len(df_features),
            'features_columns': features_col,
            'target_col': target_col,
            "target_in_the_featurs": True,
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