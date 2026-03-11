import warnings
from typing import override, Dict, Any
import logging

import pandas as pd
import awkward as ak

from sklearn.model_selection import train_test_split

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.utils.awkward_utils import ak_unflatten_discontinous_groups


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MZVAVLoader(SingleSourceLoader):
    def __init__(
        self,
        data_path,
        timestamp_name,
        target_name,
        random_state,
        group_by_days=False,
        **kwargs,
    ):

        self.data_path = data_path
        self.timestamp_name = timestamp_name
        self.target_name = target_name
        self.group_by_days = group_by_days
        self.random_state = random_state
        super().__init__(
            **kwargs,
        )

    @override
    def load_data(self):
        """Load data from the data source."""
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_splitted = True

    @override
    def get_data(self) -> dict | list[dict]:
        """Return data sources.

        Returns:
            dict | list[dict] : either dict in case we use multisource_data_splitter or list of dicts in case we use data_splitter

        """
        assert self._is_loaded is True, "Data must be loaded before getting data."
        assert self._is_splitted is True, "Data must be splitted before getting data."

        return SplitDatasetContainer(**self.data_dict)

    def _load_data(self):
        # Read raw CSV
        df, df_target, df_stamp, days_to_fault = self.read_data()

        train_val_days, test_days = train_test_split(
            days_to_fault,
            test_size=0.2,
            stratify=days_to_fault.values,
            random_state=self.random_state,
        )

        train_days, val_days = train_test_split(
            train_val_days,
            test_size=0.25,
            stratify=train_val_days.values,
            random_state=self.random_state,
        )

        split_dict = {"train": train_days, "val": val_days, "test": test_days}
        data_dict = {
            "features": df,
            "target": df_target,
            "timestamps": df_stamp,
            "initial_dates": df_stamp,
            "days": df.groupby(df.index.date).ngroup(),
        }

        out_dict = {name: {} for name, _ in data_dict.items()}
        for split_key, split_series in split_dict.items():
            for name, data in data_dict.items():
                split_mask = data.index.normalize().isin(split_series.index.normalize())
                out_dict[name][split_key] = data[split_mask].sort_index()

        if self.group_by_days:
            outs_ak = {
                k1: {
                    k2: [
                        ak_unflatten_discontinous_groups(
                            ak.from_numpy(v2.values),
                            data_dict["days"].loc[v2.index].values,
                        )
                    ]
                    for k2, v2 in v1.items()
                }
                for k1, v1 in out_dict.items()
            }
            return outs_ak
        else:
            return out_dict

    def read_data(self):
        # Read raw CSV
        df = pd.read_csv(
            self.data_path,
            sep=",",
            index_col=self.timestamp_name,
            date_format="%m/%d/%Y %H:%M",
        )

        # Fill data
        df = df.ffill()

        assert df.isna().sum().sum() == 0

        # Make labels
        fault_number_by_dates = {
            "2/12/2008": 1,
            "5/7/2008": 1,
            "5/8/2008": 1,
            "9/5/2007": 1,
            "9/6/2007": 1,  # Damper
            "8/28/2007": 2,
            "8/29/2007": 2,
            "8/30/2007": 2,  # Heating Coil
            "5/6/2008": 3,
            "8/31/2007": 3,
            "5/15/2008": 3,
            "9/1/2007": 3,
            "9/2/2007": 3,
        }  # Cooling Coil

        # fault_number_by_dates = {"2/12/2008": 1, "5/7/2008": 1, "5/8/2008": 2, "9/5/2007": 3, "9/6/2007": 4,  # Damper
        #                          "8/28/2007": 5, "8/29/2007": 6, "8/30/2007": 7,                           # Heating Coil
        #                          "5/6/2008": 8, "8/31/2007": 9, "5/15/2008": 9, "9/1/2007": 10, "9/2/2007": 11} # Cooling Coil

        for date, number in fault_number_by_dates.items():
            assert date in df.index
            df.loc[date, self.target_name] = number

        days_to_fault = df[self.target_name].resample("D").first().dropna()

        # Get timestamps
        df_stamp = df.index.to_series().copy()

        # Extract target and drop it from the dataframe and ensure to have dataframe, hence the conversion .values returns shape (-1, 1)
        df_target = df[self.target_name].to_frame().copy()
        df.drop(columns=[self.target_name], inplace=True)

        return df, df_target, df_stamp, days_to_fault

    @override
    def split_data(self):
        logger.warning(
            f"{self.data_name} does not support data splitting. The project relies on predefined splits."
        )

    def get_data_name(self):
        """Return data name."""
        return self.data_name

    def get_meta_data(self) -> Dict[str, Any]:
        """
        Return metadata about the data source.
        This can include information like data source name, version, etc.
        """
        assert self._is_loaded, "Data must be loaded before getting metadata."
        return {}
