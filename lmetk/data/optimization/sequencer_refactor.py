from typing import Optional, List, Dict, Union
import numpy as np
import torch
import awkward as ak
from torch.utils.data import Dataset
from lmetk.data.optimization.sequencer import AutoregressiveSequencer
from lmetk.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)

ArrayLike = Union[np.ndarray, torch.Tensor, ak.Array]


# ----------------------------------------------------------------------
# Base drivers
# ----------------------------------------------------------------------
class BaseSequencer:
    def __len__(self): ...
    def sequences_batch(self, index_subset): ...


class DenseDriver(BaseSequencer):
    def __init__(
        self,
        array: Union[np.ndarray, torch.Tensor],
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
    ):
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array).float()
        self.driver = AutoregressiveSequencer(
            features=array,
            time_features=None,  # no time features by default
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

    def __len__(self):
        return len(self.driver)

    def sequences_batch(self, idx):
        return self.driver.sequences_batch(idx)


class RaggedDriver(BaseSequencer):
    def __init__(
        self,
        array: ak.Array,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
        pad: bool = False,
        pad_value: float = 0.0,
    ):
        self.driver = AkwardJaggedAutoregressiveSequencer(
            features=array,
            time_features=None,  # no time features by default
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            pad=pad,
            pad_value=pad_value,
        )

    def __len__(self):
        return len(self.driver)

    def sequences_batch(self, idx):
        return self.driver.sequences_batch(idx)
