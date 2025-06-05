from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ConceptRULDataset(Dataset):
    def __init__(
        self, 
        data_split: Dict[str, np.ndarray], 
        window_size: int, 
        stride: int,
        **kwargs
    ):
        self.df_Y = data_split.rul
        self.X = data_split.features
        self.unit = data_split.unit
        self.concepts = data_split.concepts

        if "descriptors" in data_split:
            print("The dataset contains descriptors, concatenating them to the features.")
            self.X = np.concatenate((self.X, data_split.descriptors), axis=1)

        self.window_size = window_size
        self.stride = stride
        super().__init__()

    def __len__(self):
        return self.X.shape[0] // self.stride  # Floor total length divided by stride

    def __getitem__(self, i):
        # X is the normalized data

        # Sequence is entirely within the data
        if i * self.stride >= self.window_size - 1:
            unit = self.unit[
                i * self.stride - self.window_size + 1 : i * self.stride + 1
            ]  # Unit vector for desired sequence

            cond = np.where(unit != unit[0])[0]  # Index of first different value

            if cond.size == 0:  # If there is no index with different value
                i_start = i * self.stride - self.window_size + 1
                x = self.X[i_start : i * self.stride + 1, :].T
            else:
                counter = cond[0]  # Find first index of switch
                padding = (
                    self.X[i * self.stride - (self.window_size - counter) + 1]
                    .reshape(-1, 1)
                    .repeat(counter, 1)
                )
                x = self.X[
                    i * self.stride - (self.window_size - counter) + 1 : i * self.stride
                    + 1,
                    :,
                ].T
                x = np.concatenate((padding, x), 1)

            # Beginning of Sequence (backward filling)
        else:
            padding = (
                self.X[0]
                .reshape(-1, 1)
                .repeat(self.window_size - i * self.stride - 1, 1)
            )
            x = self.X[0 : i * self.stride + 1, :].T
            x = np.concatenate((padding, x), 1)

        # x = x.ravel()

        return (
            torch.Tensor(x).squeeze(),
            torch.Tensor([self.df_Y[i * self.stride]]).squeeze(),
            torch.Tensor(self.concepts[i * self.stride]),
        )