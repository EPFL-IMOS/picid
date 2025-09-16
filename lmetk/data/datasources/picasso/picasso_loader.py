import warnings

import numpy as np
import pandas as pd

from lmetk.data.datasources import SingleSourceLoader
from lmetk.data.datasources.picasso.preprocessor import sample_picasso


warnings.filterwarnings("ignore")


class PicassoLoader(SingleSourceLoader):
    def __init__(self, data_path, timestamp_name, target_name, **kwargs):

        self.data_path = data_path
        self.timestamp_name = timestamp_name
        self.target_name = target_name
        super().__init__(
            **kwargs,
        )

    def _load_data(self):
        """
        Load the data, then arrange it into a dictionary of
        tensors so it can be used by the model.
        """
        df, df_target, df_stamp = self.read_data()

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
        }

    def read_data(self):
        """
        Read the raw data from files and preprocess it.
        """
