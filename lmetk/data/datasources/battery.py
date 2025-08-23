import warnings

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from lmetk.data.datasources import SingleSourceLoader


warnings.filterwarnings("ignore")


class BatteryLoader(SingleSourceLoader):
    def __init__(self, data_path, timestamp_name, task_mode, target_name, **kwargs):
        super().__init__(
            data_path=data_path,
            timestamp_name=timestamp_name,
            task_mode=task_mode,
            target_name=target_name,
            **kwargs,
        )

    def _load_data(self):
        # Read raw CSV
        df, df_target, df_stamp = self.read_data()

        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)

        assert (
            all(mask) == False
        ), "For the datasets that potentially can have context, avoid placing the target in features, as it should be done with the preprocessing as an transformation."

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
        # data = pd.read_feather(f'{params["path"]}/{1}.feather')
        df = pd.read_csv(self.data_path, index_col=0)

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
