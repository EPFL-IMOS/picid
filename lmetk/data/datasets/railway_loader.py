import pandas as pd
import numpy as np
from omegaconf import DictConfig
from lmetk.data.datasets.abstract_classes import AbstractTimeSeriesDataSource
from lmetk.data.utils import time_features
import warnings
warnings.filterwarnings('ignore')


class RailwayLoader(AbstractTimeSeriesDataSource):
    def __init__(
        self,
        load_arguments: DictConfig,
        data_splitter,
        scaler,
    ):
        super().__init__(load_arguments, data_splitter, scaler)       
    
    
    def _load_data(self):
        # Read raw CSV
        df, df_stamp = self.read_data()
        
        target_col_name = self.load_arguments.target_name
        #  Move target column ('Bahnlast') to the front (some models require it) 
        df = pd.concat([df[[target_col_name]], df.drop(columns=[target_col_name])], axis=1)

        if self.load_arguments.trend_decomp:
            """
            Decomposes the target variable by subtracting the mean of a rolling window.
            The tagret variable then is the decomposed one. 
            """
            # Get the decomposed target variable
            S = self.load_arguments.trend_decomp_rolling_window
            self.prev_mean = df[target_col_name].rolling(S).mean().bfill()
            self.prev_mean_shifted = df[target_col_name].rolling(S).mean().shift(self.load_arguments.samples_per_day).bfill()
            decomp_target = df[target_col_name] - self.prev_mean_shifted

            df.drop(columns=[target_col_name], inplace=True)
            df.insert(loc=0, column=target_col_name, value=decomp_target)
            
        # Extract timestamps
        df_stamp, initial_dates = time_features(dates = df_stamp,
                                        use_features=self.load_arguments.time_features,
                                        normalize=self.load_arguments.normalize_time_features
                                        ) 

        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)

        return  {"features": df.values, "timestamps": df_stamp.values, "mask": mask, "initial_dates": initial_dates}

    def extract_mask(self, df: pd.DataFrame) -> np.ndarray:
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = np.array((df.columns == self.load_arguments.target_name))
        return mask
   
    def read_data(self):
        # Read raw CSV
        df = pd.read_csv(self.load_arguments.data_path, sep=",")
        
        # Get timestamps
        df_stamp = df[self.load_arguments.timestamp_name].copy()
        df.drop(columns=[self.load_arguments.timestamp_name], inplace=True)
        return df, df_stamp
        

    def _generate_splits(self):
        # Split the data (e.g., train/val/test sequences)
        train_data, val_data, test_data = self.data_splitter.split_data(self.data_dict["features"], self.data_dict["initial_dates"])
        train_timestamps, val_timestamps, test_timestamps = self.data_splitter.split_data(self.data_dict["timestamps"], self.data_dict["initial_dates"])

        # Check that splits are consistent
        assert train_data.shape[0] == train_timestamps.shape[0], "Mismatch in train data and timestamps length"
        assert val_data.shape[0] == val_timestamps.shape[0], "Mismatch in val data and timestamps length"
        assert test_data.shape[0] == test_timestamps.shape[0], "Mismatch in test data and timestamps length"

        # Update the self.data_dict
        self.data_dict.update({"features": {"train": train_data, "val": val_data, "test": test_data}})
        self.data_dict.update({"timestamps": {"train": train_timestamps, "val": val_timestamps, "test": test_timestamps}})




# if self.train_scaled is None:
#     data = pd.read_csv("datasets/railway.csv", sep=",")
#     data.datetime = pd.to_datetime(data.datetime)
#     data.set_index("datetime", inplace=True)
#     self.data = {"full": data}
    
#     samples_per_day = self.samples_per_day
    
#     if self.trend_decomp == "disabled":
#         bl = data['Bahnlast']
#         data.rename(columns={"Bahnlast": "Bahnlast_oldpos"}, inplace=True)
#         data.insert(loc=0, column='Bahnlast', value=bl)
#         data.drop(columns=['Bahnlast_oldpos'], inplace=True)
#         self.data = {"full": data, "gt": bl.copy()}
        
        
#     elif self.trend_decomp == "mean":
#         S = self.trend_decomp_rolling_window
#         self.prev_mean = data['Bahnlast'].rolling(S).mean().bfill()
#         self.prev_mean_shifted = data['Bahnlast'].rolling(S).mean().shift(samples_per_day).bfill()
        
#         decomp = data['Bahnlast'] - self.prev_mean_shifted
#         data.insert(loc=0, column='Bahnlast_decomp', value=decomp)
        
#         self.data = {"full": data, "gt": data['Bahnlast'].copy()}
#         data.drop(columns=['Bahnlast'], inplace=True)
    
#     else:
#         raise ValueError("Unknown trend decomposition")
        
#     if not self.forecast_all_features:
#         self.bahnlast_mask = (data.columns == self.forecast_feature_name)
#         self.bahnlast_mask_orig = (data.columns == self.forecast_feature_name)
#     else:
#         self.bahnlast_mask = np.ones(data.shape[1], dtype=bool)
#         self.bahnlast_mask_orig = np.ones(data.shape[1], dtype=bool)

#     test_ratio = self.test_ratio
#     val_ratio = self.val_ratio
#     train_ratio = self.train_ratio
    
#     cut_to_days = lambda x: (x // samples_per_day) * samples_per_day
#     N_samples = cut_to_days(len(data))
    
    
#     # Splitting the data
#     if self.data_split_strategy == "default":
#         N_test = cut_to_days(round(test_ratio * N_samples))
#         N_val = cut_to_days(round(val_ratio * N_samples))
#         N_train = cut_to_days(N_samples - N_test - N_val)
        
#         r_train = range(0,N_train)
#         r_val = range(N_train,N_train+N_val)
#         r_test = range(N_train+N_val,N_train+N_val+N_test)
#         r_test_small = range(data.index.get_loc("2023-01-24 00:00:00"), data.index.get_loc("2023-07-31 00:00:00"))
        
#     elif self.data_split_strategy == "scale_training_set":
#         ################################################
#         #    Train  |  Val  |  Test_small  | Test
#         #       xxxx|xxxxxxx|xxxxxxxxxxxxxxxxxxxxx
#         #   xxxxxxxx|xxxxxxx|xxxxxxxxxxxxxxxxxxxxx train 
#         #
#         ################################################
        
#         # we keep a relatively long test set
#         r_test_small = range(data.index.get_loc("2023-01-24 00:00:00"), data.index.get_loc("2023-07-31 00:00:00"))
#         r_test = range(data.index.get_loc("2023-01-24 00:00:00"), N_samples)
        
#         # we cut a small validation set
#         N_val = cut_to_days(round(val_ratio * (N_samples - len(r_test))))
#         Len_train = N_samples - N_val - len(r_test)
#         N_train = cut_to_days(round(train_ratio * (Len_train)))
            
#         r_train = range(Len_train-N_train,Len_train)
#         r_val = range(Len_train,Len_train+N_val)
        
#     elif self.data_split_strategy == "covid_training_set":
#         # we keep a relatively long test set
#         r_test_small = range(data.index.get_loc("2023-01-24 00:00:00"), data.index.get_loc("2023-07-31 00:00:00"))
#         r_test = range(data.index.get_loc("2023-01-24 00:00:00"), N_samples)
        
#         # we cut a small validation set
#         N_val = cut_to_days(round(val_ratio * (N_samples - len(r_test))))
#         Len_train = N_samples - N_val - len(r_test)
            
#         r_train = range(data.index.get_loc("2019-12-01 00:00:00"), data.index.get_loc("2021-07-31 00:00:00"))
#         r_val = range(Len_train,Len_train+N_val)

#     self.r_train, self.r_val, self.r_test, self.r_test_small = r_train, r_val, r_test, r_test_small 
#     self.train = data.iloc[r_train, :]
#     self.val = data.iloc[r_val, :]
#     self.test = data.iloc[r_test, :]
#     self.test_small = data.iloc[r_test_small, :]

    
#     # Scaling
#     scaler = MinMaxScaler()
#     scaler.fit(self.train)
#     self.train_scaled = scaler.transform(self.train)
#     self.val_scaled = scaler.transform(self.val)
#     self.test_scaled = scaler.transform(self.test)
#     self.test_small_scaled = scaler.transform(self.test_small)
    
#     if self.datetime_features:
#         dtf = time_features(
#             pd.Series(data.index),
#             use_features=["month", "day", "weekday", "hour"])
#         self.dtf_train = dtf.iloc[r_train, :]
#         self.dtf_val = dtf.iloc[r_val, :]
#         self.dtf_test = dtf.iloc[r_test, :]
#         self.dtf_test_small = dtf.iloc[r_test_small, :]
        
#         self.scaler_mask = np.concatenate(
#             (np.ones(self.train_scaled.shape[1], dtype=bool), np.zeros(4, dtype=bool)))
        
#         self.bahnlast_mask = np.concatenate(
#             (self.bahnlast_mask, np.zeros(4, dtype=bool)))
        
#         self.train_scaled = np.concatenate(
#             (self.train_scaled, self.dtf_train[['hour', 'day', 'weekday', 'month']].values), axis=1)
#         self.val_scaled = np.concatenate(
#             (self.val_scaled, self.dtf_val[['hour', 'day', 'weekday', 'month']].values), axis=1)
#         self.test_scaled = np.concatenate(
#             (self.test_scaled, self.dtf_test[['hour', 'day', 'weekday', 'month']].values), axis=1)
#         self.test_small_scaled = np.concatenate(
#             (self.test_small_scaled, self.dtf_test_small[['hour', 'day', 'weekday', 'month']].values), axis=1)
    
#     self.scaler = scaler
    
#     def to_ds(ds, step, raw_data):
        
#         out_features_mask = None
        
#         ds = ContextBatchDataset(
#             Features=torch.from_numpy(ds[:,self.bahnlast_mask]).to(torch.float32),
#             Context=torch.from_numpy(ds[:,~self.bahnlast_mask]).to(torch.float32),
#             out_features_mask=out_features_mask,
#             ts_in=self.ts_in,
#             ts_out=self.ts_out,
#             step=step,
#             raw_data=raw_data
#         )
#         return ds


#     self.datasets = dict(
#         train_test = to_ds(self.train_scaled, self.step_test, self.train),
#         train = to_ds(self.train_scaled, self.step_train, self.train),
#         val = to_ds(self.val_scaled, self.step_val, self.val),
#         test = to_ds(self.test_scaled, self.step_test, self.test),
#         test_small = to_ds(self.test_small_scaled, self.step_test, self.test)
#     )