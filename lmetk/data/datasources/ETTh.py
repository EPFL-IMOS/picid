import warnings

import numpy as np
import pandas as pd
from lmetk.data.datasources import SingleSourceLoader

from lmetk.data.data.data import DatasetChunk
import lmetk.data.data


warnings.filterwarnings("ignore")


class ETThLoader(SingleSourceLoader):
    def __init__(
        self,
        data_path: str,
        timestamp_name: str,
        task_mode: str,
        target_name: tuple[str, list],
        **kwargs,
    ):
        super().__init__(
            data_path=data_path,
            timestamp_name=timestamp_name,
            task_mode=task_mode,
            target_name=target_name,
            **kwargs,
        )

    def _load_data(self) -> dict:
        # Read raw CSV
        df, df_stamp = self.read_data()

        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)
        data_dict = {
            "features": df,
            "timestamps": df_stamp,
            "metadata":{
                "mask": mask,
                "columns": df.columns,
                "target_name": self.target_name,
            }
        }
        data = DatasetChunk(data_dict)
        return data

    def extract_mask(self, df: pd.DataFrame) -> np.ndarray:
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        if self.task_mode in ["univariate"]:
            mask = np.array((df.columns == self.target_name))
        else:
            mask = "TODO: need to be defined"

        return mask

    def read_data(
        self,
    ):
        # Read raw CSV
        df = pd.read_csv(self.data_path, sep=",")

        # Get timestamps
        df_stamp = df[self.timestamp_name].copy()
        df.drop(columns=[self.timestamp_name], inplace=True)

        # Determine columns to use
        if self.task_mode in ["univariate"]:
            df = df[[self.target_name]]

        return df, df_stamp

    def _init_default_transforms(self):
        default_transforms = getattr(self, "loader_default_transforms", {})
        return default_transforms