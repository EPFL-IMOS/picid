import warnings
import pandas as pd
from picid.data.datasources.base.single_source_loader import SingleSourceLoader

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
        self.data_path = data_path
        self.timestamp_name = timestamp_name
        self.task_mode = task_mode
        self.target_name = target_name
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

        data_dict = {
            "features": df,
            "timestamps": df_stamp,
        }

        return data_dict

    def get_meta_data(self) -> dict[str, any]:
        meta_data = super().get_meta_data()

        return {
            **meta_data,
            "target_name": self.target_name,
        }

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
