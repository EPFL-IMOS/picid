import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from lmetk.utils.timefeatures import time_features

# from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from lmetk.utils.augmentation import run_augmentation_single
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



from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig


class Dataset_ETT_hour_refactored(Dataset):
    def __init__(
        self,
        load_arguments: DictConfig,
        data_splitter,
        scaler,
    ):
        self.load_arguments = load_arguments
        self.data_splitter = data_splitter
        self.scaler = scaler

        # Core attributes extracted from config
        self.feature_mode = load_arguments.get("features", "S")
        self.target_col = load_arguments.get("target_name", "OT")
        

        self._pipeline()        
    
    def get_data(self):
        return self.splitted_data, self.splitted_timestamps

    def get_scaler(self):
        return self.scaler
    def _pipeline(self):
        # Load data
        self.data, self.timestamps = self._load_data()

        # Generate splits
        self.splitted_data, self.splitted_timestamps = self._generate_splits()

        # Apply transformations
        self._apply_transformations()
    

    def _load_data(self):
        # Read raw CSV
        df = pd.read_csv(self.load_arguments.data_path)
        self.raw_df = df
        

        # Determine columns to use
        if self.feature_mode in ['M', 'MS']:
            data_cols = df.columns.drop('date')
        else:  # 'S'
            data_cols = [self.target_col]
        
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
        

    def _generate_splits(self):
        # Split the data (e.g., train/val/test sequences)
        train_data, val_data, test_data = self.data_splitter.split_data(self.data)
        data = {"train": train_data, "val": val_data, "test": test_data}

        train_timestamps, val_timestamps, test_timestamps = self.data_splitter.split_data(self.timestamps)
        timestamps = {"train": train_timestamps, "val": val_timestamps, "test": test_timestamps}
        return data, timestamps

    def _apply_transformations(self):
        # Optionally scale the data
        if self.load_arguments.get("scale", True):
            
            self.scaler.fit(self.splitted_data["train"])
            
            for split_name in ["train", "val", "test"]:
                if split_name in self.splitted_data:
                    self.splitted_data[split_name] = self.scaler.transform(self.splitted_data[split_name])
    
    def __repr__(self) -> str:
        return f"loader_arguments: {self.load_arguments}, " \
                f"data_splitter: {self.data_splitter}, " \
                f"scaler: {self.scaler}"
            

        
        








    # def __read_data__(self):
    #     self.scaler = StandardScaler()
    #     df_raw = pd.read_csv(os.path.join(self.data_dir,
    #                                       self.data_path))

    #     border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
    #     border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    #     border1 = border1s[self.set_type]
    #     border2 = border2s[self.set_type]

    #     if self.features == 'M' or self.features == 'MS':
    #         cols_data = df_raw.columns[1:]
    #         df_data = df_raw[cols_data]
        
    #     elif self.features == 'S':
    #         df_data = df_raw[[self.target]]

    #     if self.scale:
    #         train_data = df_data[border1s[0]:border2s[0]]
    #         self.scaler.fit(train_data.values)
    #         data = self.scaler.transform(df_data.values)
    #     else:
    #         data = df_data.values

    #     df_stamp = df_raw[['date']][border1:border2]
    #     df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
    #     if self.timeenc == 0:
    #         df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    #         df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    #         df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    #         df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    #         data_stamp = df_stamp.drop(columns=['date']).values
       
    #     elif self.timeenc == 1:
    #         data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
    #         data_stamp = data_stamp.transpose(1, 0) 

    #     self.data_x = data[border1:border2]
    #     self.data_y = data[border1:border2]

    #     # if self.set_type == 0 and self.augmentation_ratio > 0:
    #     #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

    #     self.data_stamp = data_stamp
    
    

    # def inverse_transform(self, data):
    #     return self.scaler.inverse_transform(data)

    
