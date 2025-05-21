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


from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig


class RailwayLoader:
    def __init__(
        self,
        load_arguments: DictConfig,
        data_splitter,
        scaler,
    ):
        self.load_arguments = load_arguments
        self.data_splitter = data_splitter
        self.scaler = scaler

        # # Core attributes extracted from config
        # self.feature_mode = load_arguments.get("features", "S")
        # self.target_col = load_arguments.get("target_name", "OT")
        

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
        
        target_col_name = self.load_arguments.target_col_name
        # Read raw CSV
        # data_path = "datasets/railway.csv"
        df = pd.read_csv(self.load_arguments.data_path, sep=",")
        self.raw_df = df.copy()
        
        samples_per_day = 24

        # Move target column to the front (some models require it)
        
        
        #  Move target column ('Bahnlast') to the front (some models require it) 
        df = pd.concat([df[[target_col_name]], df.drop(columns=[target_col_name])], axis=1)
        
        # Comment for now
        # # Determine columns to use
        # if self.feature_mode in ['M', 'MS']:
        #     data_cols = df.columns.drop('date')
        # else:  # 'S'
        #     data_cols = [self.target_col]
        
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





if self.train_scaled is None:
    data = pd.read_csv("datasets/railway.csv", sep=",")
    data.datetime = pd.to_datetime(data.datetime)
    data.set_index("datetime", inplace=True)
    self.data = {"full": data}
    
    samples_per_day = self.samples_per_day
    
    if self.trend_decomp == "disabled":
        bl = data['Bahnlast']
        data.rename(columns={"Bahnlast": "Bahnlast_oldpos"}, inplace=True)
        data.insert(loc=0, column='Bahnlast', value=bl)
        data.drop(columns=['Bahnlast_oldpos'], inplace=True)
        self.data = {"full": data, "gt": bl.copy()}
        
        
    elif self.trend_decomp == "mean":
        S = self.trend_decomp_rolling_window
        self.prev_mean = data['Bahnlast'].rolling(S).mean().bfill()
        self.prev_mean_shifted = data['Bahnlast'].rolling(S).mean().shift(samples_per_day).bfill()
        
        decomp = data['Bahnlast'] - self.prev_mean_shifted
        data.insert(loc=0, column='Bahnlast_decomp', value=decomp)
        
        self.data = {"full": data, "gt": data['Bahnlast'].copy()}
        data.drop(columns=['Bahnlast'], inplace=True)
    else:
        raise ValueError("Unknown trend decomposition")
        
    if not self.forecast_all_features:
        self.bahnlast_mask = (data.columns == self.forecast_feature_name)
        self.bahnlast_mask_orig = (data.columns == self.forecast_feature_name)
    else:
        self.bahnlast_mask = np.ones(data.shape[1], dtype=bool)
        self.bahnlast_mask_orig = np.ones(data.shape[1], dtype=bool)

    test_ratio = self.test_ratio
    val_ratio = self.val_ratio
    train_ratio = self.train_ratio
    
    cut_to_days = lambda x: (x // samples_per_day) * samples_per_day
    N_samples = cut_to_days(len(data))
    
    
    # Splitting the data
    if self.data_split_strategy == "default":
        N_test = cut_to_days(round(test_ratio * N_samples))
        N_val = cut_to_days(round(val_ratio * N_samples))
        N_train = cut_to_days(N_samples - N_test - N_val)
        
        r_train = range(0,N_train)
        r_val = range(N_train,N_train+N_val)
        r_test = range(N_train+N_val,N_train+N_val+N_test)
        r_test_small = range(data.index.get_loc("2023-01-24 00:00:00"), data.index.get_loc("2023-07-31 00:00:00"))
        
    elif self.data_split_strategy == "scale_training_set":
        ################################################
        #    Train  |  Val  |  Test_small  | Test
        #       xxxx|xxxxxxx|xxxxxxxxxxxxxxxxxxxxx
        #   xxxxxxxx|xxxxxxx|xxxxxxxxxxxxxxxxxxxxx train 
        #
        ################################################
        
        # we keep a relatively long test set
        r_test_small = range(data.index.get_loc("2023-01-24 00:00:00"), data.index.get_loc("2023-07-31 00:00:00"))
        r_test = range(data.index.get_loc("2023-01-24 00:00:00"), N_samples)
        
        # we cut a small validation set
        N_val = cut_to_days(round(val_ratio * (N_samples - len(r_test))))
        Len_train = N_samples - N_val - len(r_test)
        N_train = cut_to_days(round(train_ratio * (Len_train)))
            
        r_train = range(Len_train-N_train,Len_train)
        r_val = range(Len_train,Len_train+N_val)
        
    elif self.data_split_strategy == "covid_training_set":
        # we keep a relatively long test set
        r_test_small = range(data.index.get_loc("2023-01-24 00:00:00"), data.index.get_loc("2023-07-31 00:00:00"))
        r_test = range(data.index.get_loc("2023-01-24 00:00:00"), N_samples)
        
        # we cut a small validation set
        N_val = cut_to_days(round(val_ratio * (N_samples - len(r_test))))
        Len_train = N_samples - N_val - len(r_test)
            
        r_train = range(data.index.get_loc("2019-12-01 00:00:00"), data.index.get_loc("2021-07-31 00:00:00"))
        r_val = range(Len_train,Len_train+N_val)

    self.r_train, self.r_val, self.r_test, self.r_test_small = r_train, r_val, r_test, r_test_small 
    self.train = data.iloc[r_train, :]
    self.val = data.iloc[r_val, :]
    self.test = data.iloc[r_test, :]
    self.test_small = data.iloc[r_test_small, :]

    
    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(self.train)
    self.train_scaled = scaler.transform(self.train)
    self.val_scaled = scaler.transform(self.val)
    self.test_scaled = scaler.transform(self.test)
    self.test_small_scaled = scaler.transform(self.test_small)
    
    if self.datetime_features:
        dtf = time_features(
            pd.Series(data.index),
            use_features=["month", "day", "weekday", "hour"])
        self.dtf_train = dtf.iloc[r_train, :]
        self.dtf_val = dtf.iloc[r_val, :]
        self.dtf_test = dtf.iloc[r_test, :]
        self.dtf_test_small = dtf.iloc[r_test_small, :]
        
        self.scaler_mask = np.concatenate(
            (np.ones(self.train_scaled.shape[1], dtype=bool), np.zeros(4, dtype=bool)))
        
        self.bahnlast_mask = np.concatenate(
            (self.bahnlast_mask, np.zeros(4, dtype=bool)))
        
        self.train_scaled = np.concatenate(
            (self.train_scaled, self.dtf_train[['hour', 'day', 'weekday', 'month']].values), axis=1)
        self.val_scaled = np.concatenate(
            (self.val_scaled, self.dtf_val[['hour', 'day', 'weekday', 'month']].values), axis=1)
        self.test_scaled = np.concatenate(
            (self.test_scaled, self.dtf_test[['hour', 'day', 'weekday', 'month']].values), axis=1)
        self.test_small_scaled = np.concatenate(
            (self.test_small_scaled, self.dtf_test_small[['hour', 'day', 'weekday', 'month']].values), axis=1)
    
    self.scaler = scaler
    
    def to_ds(ds, step, raw_data):
        
        out_features_mask = None
        
        ds = ContextBatchDataset(
            Features=torch.from_numpy(ds[:,self.bahnlast_mask]).to(torch.float32),
            Context=torch.from_numpy(ds[:,~self.bahnlast_mask]).to(torch.float32),
            out_features_mask=out_features_mask,
            ts_in=self.ts_in,
            ts_out=self.ts_out,
            step=step,
            raw_data=raw_data
        )
        return ds


    self.datasets = dict(
        train_test = to_ds(self.train_scaled, self.step_test, self.train),
        train = to_ds(self.train_scaled, self.step_train, self.train),
        val = to_ds(self.val_scaled, self.step_val, self.val),
        test = to_ds(self.test_scaled, self.step_test, self.test),
        test_small = to_ds(self.test_small_scaled, self.step_test, self.test)
    )