import pandas as pd
import numpy as np
from omegaconf import DictConfig
from lmetk.data.utils import time_features as railway_timefeatures
from lmetk.data.timefeatures import time_features as timepfn_timefeatures
from lmetk.data.datasources.base import AbstractTimeSeriesDataSource
import warnings

warnings.filterwarnings("ignore")


class ETThLoader(AbstractTimeSeriesDataSource):
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
        if self.load_arguments.features_mode in ["S"]:
            df = df[[self.load_arguments.target_name]]

        # Extract time features
        if self.load_arguments.timeenc == 'railway_timefeatures':
            df_stamp, self.initial_dates = railway_timefeatures(
                dates=df_stamp,
                use_features=self.load_arguments.time_features,
                normalize=self.load_arguments.normalize_time_features,
            )
        elif self.load_arguments.timeenc == 'timepfn_timefeatures':
            self.initial_dates = pd.DataFrame(pd.to_datetime(df_stamp))
            df_stamp = timepfn_timefeatures(pd.to_datetime(self.initial_dates['date'].values), freq=self.load_arguments.freq)
            df_stamp = pd.DataFrame(df_stamp.transpose(1, 0))
        else:
            raise ValueError(f"Unknown time feature encoding : load_arguments.timeenc={self.load_arguments.timeenc}")

        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)

        return {"features": df.values, "timestamps": df_stamp.values, "mask": mask}

    def extract_mask(self, df: pd.DataFrame) -> np.ndarray:
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = np.array((df.columns == self.load_arguments.target_name))
        return mask

    def read_data(
        self,
    ):
        # Read raw CSV
        df = pd.read_csv(self.load_arguments.data_path, sep=",")

        # Get timestamps
        df_stamp = df[self.load_arguments.timestamp_name].copy()
        df.drop(columns=[self.load_arguments.timestamp_name], inplace=True)
        return df, df_stamp
