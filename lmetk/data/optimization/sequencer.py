from typing import Union

import awkward as ak
import numpy as np
import torch

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
        self.driver = AkwardJaggedAutoregressiveSequencer(
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
