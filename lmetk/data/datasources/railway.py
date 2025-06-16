import pandas as pd
import numpy as np
from omegaconf import DictConfig
from lmetk.data.datasources.base import DataSourceLoader
from lmetk.data.utils import time_features
import warnings

warnings.filterwarnings("ignore")


class RailwayLoader(DataSourceLoader):
    def __init__(
        self,
        data_path,
        timestamp_name,
        task_mode,
        target_name,
        **kwargs
    ):
        super().__init__(
            data_path=data_path, 
            timestamp_name=timestamp_name, 
            task_mode=task_mode,
            target_name=target_name,
            **kwargs
            )

    def _load_data(self):
        # Read raw CSV
        df, df_target, df_stamp = self.read_data()        

        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)
        
        assert all(mask) == False, "For the datasets that potentially can have context, avoid placing the target in features, as it should be done with the preprocessing as an transformation."

        return {
            "features": df,
            "target": df_target,
            "timestamps": df_stamp,
            "mask": mask,
            "initial_dates": df_stamp,
        }

    def extract_mask(self, df: pd.DataFrame) -> np.ndarray:
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = np.array((df.columns == self.target_name))
        return mask

    def read_data(self):
        # Read raw CSV
        df = pd.read_csv(self.data_path, sep=",")

        # Get timestamps
        df_stamp = df[self.timestamp_name].copy()
        df.drop(columns=[self.timestamp_name], inplace=True)

        # Extract target and drop it from the dataframe and ensure to have dataframe, hence the conversion .values returns shape (-1, 1)
        df_target = pd.DataFrame(df[self.target_name].copy())
        df.drop(columns=[self.target_name], inplace=True)

        # #  Move target column ('Bahnlast') to the front (some models require it)
        # df = pd.concat(
        #     [df[[self.target_name]], df.drop(columns=[self.target_name])], axis=1
        # )

        return df, df_target, df_stamp



# if self.trend_decomp:
#     """
#     Decomposes the target variable by subtracting the mean of a rolling window.
#     The tagret variable then is the decomposed one. 
#     """
#     # Get the decomposed target variable
#     S = self.trend_decomp_rolling_window
#     self.prev_mean = df[target_col_name].rolling(S).mean().bfill()
#     self.prev_mean_shifted = (
#         df[target_col_name]
#         .rolling(S)
#         .mean()
#         .shift(self.samples_per_day)
#         .bfill()
#     )
#     decomp_target = df[target_col_name] - self.prev_mean_shifted

#     df.drop(columns=[target_col_name], inplace=True)
#     df.insert(loc=0, column=target_col_name, value=decomp_target)


    # def _generate_splits(self):
    #     # Split the data (e.g., train/val/test sequences)
    #     train_data, val_data, test_data = self.data_splitter.split_data(
    #         self.data_dict["features"], self.data_dict["initial_dates"]
    #     )
    #     train_timestamps, val_timestamps, test_timestamps = (
    #         self.data_splitter.split_data(
    #             self.data_dict["timestamps"], self.data_dict["initial_dates"]
    #         )
    #     )

    #     # Check that splits are consistent
    #     assert train_data.shape[0] == train_timestamps.shape[0], (
    #         "Mismatch in train data and timestamps length"
    #     )
    #     assert val_data.shape[0] == val_timestamps.shape[0], (
    #         "Mismatch in val data and timestamps length"
    #     )
    #     assert test_data.shape[0] == test_timestamps.shape[0], (
    #         "Mismatch in test data and timestamps length"
    #     )

    #     # Update the self.data_dict
    #     self.data_dict.update(
    #         {"features": {"train": train_data, "val": val_data, "test": test_data}}
    #     )
    #     self.data_dict.update(
    #         {
    #             "timestamps": {
    #                 "train": train_timestamps,
    #                 "val": val_timestamps,
    #                 "test": test_timestamps,
    #             }
    #         }
    #     )

    #     self._is_split = True  # Mark data as split
