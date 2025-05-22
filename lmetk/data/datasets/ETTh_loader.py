import pandas as pd
import numpy as np
from omegaconf import DictConfig
from lmetk.data.utils import time_features
from lmetk.data.datasets.abstract_classes import AbstractTimeSeriesDataSource

import warnings
warnings.filterwarnings('ignore')



# default_load_arguments = {
#     "data_name": "ETTh1",
#     "data_folder": "ETT-small",
#     "data_dir": "${paths.data_dir}/${dataset.loader.data_folder}",  # interpolated by Hydra
#     "flag": "train",
#     "size": None,
#     "features": "S",
#     "target_name": "OT",
#     "scale": True,
#     "timeenc": 0,
#     "freq": "h"
# }


class Dataset_ETT_hour_refactored(AbstractTimeSeriesDataSource):
    def __init__(
        self,
        load_arguments: DictConfig,
        data_splitter,
        scaler,
    ):

        super().__init__(load_arguments, data_splitter, scaler)
        
    
    def _load_data(self) -> dict:
        # Read raw CSV
        df, df_stamp = self.read_data()

        # Determine columns to use
        if self.load_arguments.features_mode in ['S']:
            df = df[[self.load_arguments.target_name]] 

        
        # Extract time features
        df_stamp, self.initial_dates = time_features(dates = df_stamp,
                                        use_features=self.load_arguments.time_features,
                                        normalize=self.load_arguments.normalize_time_features
                                        ) 
        
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)

        return {"features": df.values, "timestamps": df_stamp.values, "mask": mask}

    def extract_mask(self, df: pd.DataFrame) -> np.ndarray:
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = np.array((df.columns == self.load_arguments.target_name))
        return mask

    def read_data(self,):
        # Read raw CSV
        df = pd.read_csv(self.load_arguments.data_path, sep=",")
        
        # Get timestamps
        df_stamp = df[self.load_arguments.timestamp_name].copy()
        df.drop(columns=[self.load_arguments.timestamp_name], inplace=True)
        return df, df_stamp
        
