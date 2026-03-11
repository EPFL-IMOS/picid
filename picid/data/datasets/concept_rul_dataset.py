import numpy as np
from numpy import ndarray

import torch
from picid.data.datasets.base import BaseDataset
from lightning_fabric.utilities.data import AttributeDict

import logging
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)


class ConceptRULDataset(BaseDataset):
    def __init__(
        self,
        data_dict: dict[str, ndarray | list[ndarray]],
        window_size: int,
        stride: int,
        **kwargs,
    ):
        for k, v in data_dict.items():
            assert isinstance(v, (np.ndarray)), (
                f"All entries in data_dict must be numpy arrays "
                f"or lists of numpy arrays. Found {type(v)} for key {k}."
            )

        self.window_size = window_size
        self.stride = stride

        self.df_Y = data_dict.rul
        self.X = data_dict.features

        self.timestamps = data_dict.timestamps
        self.health_states = data_dict.health_states
        # self.descriptors = data_dict.descriptors
        self.concepts = data_dict.concepts
        self.unit = data_dict.unit

        # INFO: We have moved this to the preprocessing transforms
        # if getattr(self, "descriptors", None) is not None:
        #     logger.info(
        #         "The dataset contains descriptors, concatenating them to the features."
        #     )
        #     self.X = np.concatenate((self.X, data_dict["descriptors"]), axis=1)

        super().__init__(data_dict, **kwargs)

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

        # We change x back to (time, features) format in this project
        x = x.T

        return AttributeDict(
            {
                "features": torch.Tensor(x).squeeze(),
                "rul": torch.Tensor([self.df_Y[i * self.stride]]).reshape(-1, 1),
                "concepts": torch.Tensor(self.concepts[i * self.stride]),
                "health_states": torch.Tensor(
                    [self.health_states[i * self.stride]]
                ).squeeze(),
            }
        )

    def get_collate_fn(self):
        return default_collate
