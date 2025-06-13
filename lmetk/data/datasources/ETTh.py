import pandas as pd
import numpy as np
from omegaconf import DictConfig
from lmetk.data.utils import time_features as railway_timefeatures
from lmetk.data.timefeatures import time_features as timepfn_timefeatures
from lmetk.data.datasources.base import DataSourceLoader
import warnings



warnings.filterwarnings("ignore")

class ETThLoader(DataSourceLoader):
    def __init__(
        self,
        data_path: str,
        timestamp_name: str,
        task_mode: str, 
        target_name: tuple[str,list], 
        **kwargs
    ):
        super().__init__(
            data_path=data_path, 
            timestamp_name=timestamp_name, 
            task_mode=task_mode,
            target_name=target_name,
            **kwargs
            )
    
    def _load_data(self) -> dict:
        # Read raw CSV
        df, df_stamp = self.read_data()
        
        # Extract the mask of target, i.e., a mask where all ones are the target the rest is zero
        mask = self.extract_mask(df)

        return {
            "features": df, 
            "timestamps": df_stamp, 
            "mask": mask, 
            "columns": df.columns, 
            "target_name": self.target_name,
            "metadata": None, 
            }

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
        default_transforms = getattr(self, 'loader_default_transforms', {})
        return default_transforms
       
       
        # This has to go default loader transforms
        # # Extract time features
        # if self.load_arguments.timeenc == 'railway_timefeatures':
        #     df_stamp, self.initial_dates = railway_timefeatures(
        #         dates=df_stamp,
        #         use_features=self.load_arguments.time_features,
        #         normalize=self.load_arguments.normalize_time_features,
        #     )
        # elif self.load_arguments.timeenc == 'timepfn_timefeatures':
        #     self.initial_dates = pd.DataFrame(pd.to_datetime(df_stamp))
        #     df_stamp = timepfn_timefeatures(pd.to_datetime(self.initial_dates['date'].values), freq=self.load_arguments.freq)
        #     df_stamp = pd.DataFrame(df_stamp.transpose(1, 0))
        # else:
        #     raise ValueError(f"Unknown time feature encoding : load_arguments.timeenc={self.load_arguments.timeenc}")
