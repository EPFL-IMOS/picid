# This was a prototype! Needs work.

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from picid.data.datasources.base.single_source_loader import SingleSourceLoader

logger = logging.getLogger(__name__)


def subsampling(df, subsampling_rate):
    """
    reduce computational cost by subsampling the data
    :param df: pd.pd.DataFrame, subsampling data
    :param subsampling_rate: int, subsampling rate, reduce size to 1/subsampling_rate
    """
    return df[::subsampling_rate]


def binarize_concept(x):
    return x < -0.0015


def scale_concept(x):
    return np.clip(x / -0.035, 0, 1)


class AGTF30KDataSource(SingleSourceLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_data(self):
        # Load the data from the specified path
        self.mode = self.mode
        self.window_size = self.window_size
        self.stride = self.stride

        # if self.mode == "train":
        #     filename = os.path.join(
        #     Path(self.path), f"_test_DT.csv"
        # )
        # elif self.mode == "val":
        #     filename = os.path.join(
        #     Path(self.path), f"_val_SV.csv"
        # )
        # elif self.mode == "test":
        #     filename = os.path.join(
        #     Path(self.path), f"_test_ST.csv"
        # )

        filename_train = os.path.join(Path(self.path), "_train_DT.csv")
        if not os.path.exists(filename_train):
            raise FileNotFoundError(f"Train file not found: {filename_train}")

        filename_test = os.path.join(Path(self.path), "_test_DT.csv")
        if not os.path.exists(filename_test):
            raise FileNotFoundError(f"Test file not found: {filename_test}")

        filename_val = os.path.join(Path(self.path), "_val_DT.csv")
        if not os.path.exists(filename_val):
            raise FileNotFoundError(f"Validation file not found: {filename_val}")

        df_train = pd.read_csv(filename_train, index_col=0)
        df_test = pd.read_csv(filename_test, index_col=0)
        df_val = pd.read_csv(filename_val, index_col=0)

        # Concatenate all dataframes
        df = pd.concat([df_train, df_test, df_val], axis=0)

        # timestamps
        timestamps = df["signaldate"].copy()

        # operating conditions: 'alt', 'MN', 'PLA'
        W_var = ["alt", "MN", "PLA"]
        self.df_W = df[W_var]

        # sensor readings
        XS_VAR = [
            "Wf",
            "Pa",
            "S2_Pt",
            "S25_Pt",
            "S36_Pt",
            "S45_Tt",
            "S5_Pt",
            "VAFN",
            "N_LPC",
            "N_HPC",
        ]
        self.df_X = df[XS_VAR]

        # fault_type
        # in AGTF30K, there are 17 types of faults
        # 0 means no fault (healthy)
        self.df_Y = df[["V"]]

        if self.subsampling_rate > 1:
            # subsample
            self.df_A = subsampling(self.df_A, self.subsampling_rate)
            self.df_W = subsampling(self.df_W, self.subsampling_rate)
            self.df_X = subsampling(self.df_X, self.subsampling_rate)
            self.df_Y = subsampling(self.df_Y, self.subsampling_rate)
            timestamps = subsampling(timestamps, self.subsampling_rate)

        # remove non-minimum degradation
        # self.concepts = self.concepts.apply(lambda row: pd.Series([val if val == min(row) else 0 for val in row]), axis=1)

        # drop constant concepts
        # self.concepts = self.concepts.loc[:, (self.concepts != self.concepts.iloc[0]).any()]
        logger.info(f"Used concepts: {self.concepts.columns}")
        self.concepts = self.concepts.astype(int)

        return {
            "features": self.df_X.values,  # sensor data measurements
            "timestamps": timestamps.values,
            "descriptors": self.df_W.values,  # operating conditions
            "fault-type": self.df_Y.values,  # fault type
            # degradation=self.df_T.values, # degradation resp. concepts
            # auxiliary_data=self.df_A.values, # aux data: 'unit', 'cycle', 'Fc', 'hs'
        }
