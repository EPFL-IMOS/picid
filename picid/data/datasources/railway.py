import warnings

import numpy as np
import pandas as pd

from picid.data.datasources.base.single_source_loader import SingleSourceLoader


warnings.filterwarnings("ignore")


class RailwayLoader(SingleSourceLoader):
    def __init__(self, data_path, timestamp_name, target_name, **kwargs):

        self.data_path = data_path
        self.timestamp_name = timestamp_name
        self.target_name = target_name
        super().__init__(
            **kwargs,
        )

    def _load_data(self):
        # Read raw CSV
        df, df_target, df_stamp = self.read_data()
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        self.mask = self.extract_mask(df)

        assert (
            all(self.mask) == False
        ), "For the datasets that potentially can have context, avoid placing the target in features, as it should be done with the preprocessing as an transformation."

        return {
            "features": df,
            "target": df_target,
            "timestamps": df_stamp,
            "initial_dates": df_stamp,
        }

    def get_meta_data(self) -> dict[str, any]:
        meta_data = super().get_meta_data()

        return {
            **meta_data,
            "mask": self.mask,
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

        return df, df_target, df_stamp
