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


class Dataset_ETT_hour(Dataset):
    def __init__(self,
                data_dir, 
                data_name='ETTh1',
                flag='train',
                size=None,
                features='S',
                target_name='OT',
                scale=True,
                timeenc=0,
                freq='h',
                augmentation_ratio=0,
                 **args, 
                ):

        # Things I got rid of:
        # data_path='ETTh1.csv',
        # ------------

        # size [seq_len, label_len, pred_len]
        self.args = args
        
        # # info
        # if size == None:
        #     self.seq_len = 24 * 4 * 4
        #     self.label_len = 24 * 4
        #     self.pred_len = 24 * 4
        # else:
        #     self.seq_len = size[0]
        #     self.label_len = size[1]
        #     self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target_name
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.augmentation_ratio = augmentation_ratio
        self.data_dir = data_dir
        self.data_path = f'{data_name}.csv'
        self.__read_data__()

    def _load_data(self):
        df = pd.read_csv(self.data_path)
        self.raw_df = df

        if self.feature_mode in ['M', 'MS']:
            data_cols = df.columns[1:]  # skip 'date'
        else:  # 'S'
            data_cols = [self.target_col]
        
        data = df[data_cols]
        self.data = data.values

        self.dates = pd.to_datetime(df['date'])

        n = len(df)
        n_train = 12 * 30 * 24 - self.seq_len #int(n * self.train_ratio)
        n_val = #int(n * self.val_ratio)
        n_test = #n - n_train - n_val

        self.splits = {
            'train': (0, n_train),
            'val': (n_train, n_train + n_val),
            'test': (n_train + n_val, n)
        }

        # Fit scaler on train data
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(self.data[self.splits['train'][0]:self.splits['train'][1]])
            self.data = self.scaler.transform(self.data)

        self.time_features = self._generate_time_features()










    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.data_dir,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=['date']).values
       
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp
    
    def get_split(self, split_name):
        i1, i2 = self.splits[split_name]
        return self.data[i1:i2], self.time_features[i1:i2]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def __repr__(self) -> str:
        return f"data_name: {self.data_path}, " \
               f"seq_len: {self.seq_len}, " \
               f"label_len: {self.label_len}, " \
               f"pred_len: {self.pred_len}, " \
               f"features: {self.features}, " \
               f"target: {self.target}, " \
               f"scale: {self.scale}, " \
               f"timeenc: {self.timeenc}, " \
               f"freq: {self.freq}"