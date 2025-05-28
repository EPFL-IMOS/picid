import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from itertools import combinations


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


def flatten_RUL(df):
    hs_change = df[df["hs"].diff() == -1]
    if len(hs_change) > 0:
        df["RUL"].loc[df["cycle"] < hs_change.iloc[0]["cycle"]] = hs_change.iloc[0][
            "RUL"
        ]
    return df


class NCMAPSSDataset(Dataset):
    def __init__(
        self,
        path,
        n_DS="02",
        mode="train",
        cruise=False,
        units=None,
        subsampling_rate=1,
        concepts=["LPT", "HPT"],
        combined_concepts=False,
        binary_concepts=True,
        RUL="linear",
        include_healthy=True,
        window_size=50,
        stride=1,
        scaling="legacy",
        **kwargs,
    ):

        self.df_Y /= 100.0


        # available features: measurements (X) + descriptors (W)
        self.X = pd.concat((self.df_X, self.df_W), axis=1)
        del self.df_X, self.df_W

    def __len__(self):
        return self.X.shape[0] // self.stride  # Floor total length divided by stride

    def __getitem__(self, i):
        # Sequence is entirely within the data
        if i * self.stride >= self.window_size - 1:
            unit = self.df_A["unit"].iloc[
                i * self.stride - self.window_size + 1 : i * self.stride + 1
            ]  # Unit vector for desired sequence

            cond = np.where(unit != unit.iloc[0])[0]  # Index of first different value

            if cond.size == 0:  # If there is no index with different value
                i_start = i * self.stride - self.window_size + 1
                x = self.X.iloc[i_start : i * self.stride + 1, :].values.T
            else:
                counter = cond[0]  # Find first index of switch
                padding = (
                    self.X.iloc[i * self.stride - (self.window_size - counter) + 1]
                    .values.reshape(-1, 1)
                    .repeat(counter, 1)
                )
                x = self.X.iloc[
                    i * self.stride - (self.window_size - counter) + 1 : i * self.stride
                    + 1,
                    :,
                ].values.T
                x = np.concatenate((padding, x), 1)

        # Beginning of Sequence (backward filling)
        else:
            padding = (
                self.X.iloc[0]
                .values.reshape(-1, 1)
                .repeat(self.window_size - i * self.stride - 1, 1)
            )
            x = self.X.iloc[0 : i * self.stride + 1, :].values.T
            x = np.concatenate((padding, x), 1)

        return (
            torch.Tensor(x),
            torch.Tensor(self.df_Y.iloc[i * self.stride]).squeeze(),
            torch.Tensor(self.concepts.iloc[i * self.stride].to_numpy()),
        )
