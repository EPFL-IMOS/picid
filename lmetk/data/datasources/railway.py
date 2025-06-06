import pandas as pd
import numpy as np
from omegaconf import DictConfig
from lmetk.data.datasources.base import AbstractTimeSeriesDataSource
from lmetk.data.utils import time_features
import warnings

warnings.filterwarnings("ignore")


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
        df = pd.concat(
            [df[[target_col_name]], df.drop(columns=[target_col_name])], axis=1
        )

        if self.load_arguments.trend_decomp:
            """
            Decomposes the target variable by subtracting the mean of a rolling window.
            The tagret variable then is the decomposed one. 
            """
            # Get the decomposed target variable
            S = self.load_arguments.trend_decomp_rolling_window
            self.prev_mean = df[target_col_name].rolling(S).mean().bfill()
            self.prev_mean_shifted = (
                df[target_col_name]
                .rolling(S)
                .mean()
                .shift(self.load_arguments.samples_per_day)
                .bfill()
            )
            decomp_target = df[target_col_name] - self.prev_mean_shifted

            df.drop(columns=[target_col_name], inplace=True)
            df.insert(loc=0, column=target_col_name, value=decomp_target)

        # Extract timestamps
        df_stamp, initial_dates = time_features(
            dates=df_stamp,
            use_features=self.load_arguments.time_features,
            normalize=self.load_arguments.normalize_time_features,
        )

        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)
        super()._load_data()
        
        return {
            "features": df.values,
            "timestamps": df_stamp.values,
            "mask": mask,
            "initial_dates": initial_dates,
        }

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
        train_data, val_data, test_data = self.data_splitter.split_data(
            self.data_dict["features"], self.data_dict["initial_dates"]
        )
        train_timestamps, val_timestamps, test_timestamps = (
            self.data_splitter.split_data(
                self.data_dict["timestamps"], self.data_dict["initial_dates"]
            )
        )

        # Check that splits are consistent
        assert train_data.shape[0] == train_timestamps.shape[0], (
            "Mismatch in train data and timestamps length"
        )
        assert val_data.shape[0] == val_timestamps.shape[0], (
            "Mismatch in val data and timestamps length"
        )
        assert test_data.shape[0] == test_timestamps.shape[0], (
            "Mismatch in test data and timestamps length"
        )

        # Update the self.data_dict
        self.data_dict.update(
            {"features": {"train": train_data, "val": val_data, "test": test_data}}
        )
        self.data_dict.update(
            {
                "timestamps": {
                    "train": train_timestamps,
                    "val": val_timestamps,
                    "test": test_timestamps,
                }
            }
        )

        self._is_split = True  # Mark data as split
