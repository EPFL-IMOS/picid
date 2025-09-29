from typing import Union

import awkward as ak
import numpy as np
import torch

from lmetk.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from lmetk.transforms.base.multisource import find_singular_ragged_dim
import logging

ArrayLike = Union[np.ndarray, torch.Tensor, ak.Array]
# Set up a logger for better feedback
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Base drivers
# ----------------------------------------------------------------------
class BaseSequencer:
    def __len__(self): ...
    def sequences_batch(self, index_subset): ...


class DenseArraySequencer(BaseSequencer):
    def __init__(
        self,
        array: Union[np.ndarray, torch.Tensor],
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
    ):
        if isinstance(array, np.ndarray):
            array = ak.Array(array)
        elif isinstance(array, torch.Tensor):
            array = ak.Array(array.numpy())
        else:
            raise TypeError(
                f"array must be np.ndarray or torch.Tensor but got {type(array)}."
            )
        array = ak.values_astype(array, np.float32)
        array = ak.from_regular(array, axis=0)  # ensure regularity
        self.driver = AkwardJaggedAutoregressiveSequencer(
            features=array,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

    def __len__(self):
        return len(self.driver)

    def get_index_array(self):
        return self.driver.get_index_array()

    def sequences_batch(self, idx):
        return self.driver.sequences_batch(idx)


class RaggedArraySequencer(BaseSequencer):
    def __init__(
        self,
        array: ak.Array,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
    ):
        array = ak.values_astype(array, np.float32)
        ragged_dim = find_singular_ragged_dim({"array": array})  # validate ragged dim
        if ragged_dim is None:
            array = ak.from_regular(array, axis=0)  # ensure regularity
            # Creates a lot of noise in the logs, some way to reduce it?
            # logger.warning("Input array do not have ragged dim.")

        self.driver = AkwardJaggedAutoregressiveSequencer(
            features=array,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

    def __len__(self):
        return len(self.driver)

    def get_index_array(self):
        return self.driver.get_index_array()

    def sequences_batch(self, idx):
        return self.driver.sequences_batch(idx)
