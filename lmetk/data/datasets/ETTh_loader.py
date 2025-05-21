import pandas as pd

from torch.utils.data import Dataset

from lmetk.utils.timefeatures import time_features

from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig

import warnings
from lmetk.data.datasets.abstract_classes import AbstractTimeSeriesDataSource
from omegaconf import DictConfig
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
        
    
    def _load_data(self):
        # Read raw CSV
        df = pd.read_csv(self.load_arguments.data_path)
        self.raw_df = df
        

        # Determine columns to use
        if self.load_arguments.features_mode in ['M', 'MS']:
            data_cols = df.columns.drop('date')
        else:  # 'S'
            data_cols = [self.load_arguments.target_name]
        
        # Extract timestamps
        df_stamp = self.raw_df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.load_arguments.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp = df_stamp.drop(columns=['date']) #.values
        
        elif self.load_arguments.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 


        # Extract data
        return df[data_cols].values, df_stamp.values
        
