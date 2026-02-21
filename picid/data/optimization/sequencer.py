from typing import Union

import awkward as ak
import numpy as np
import pandas as pd
import torch

from picid.data.optimization.ak_jagged_sequencer import (
    AkwardJaggedAutoregressiveSequencer,
)
from picid.utils.awkward_utils import find_singular_ragged_dim
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
        pred_offset: int = 0,
        padding_left_flag: Union[bool, int] = True,
        warmup_steps: int = None,
    ):
        if isinstance(array, np.ndarray):
            array = ak.Array(array)
        elif isinstance(array, torch.Tensor):
            array = ak.Array(array.numpy())
        elif isinstance(array, pd.DataFrame):
            array = ak.Array(array.to_numpy())
        else:
            raise TypeError(
                f"array must be np.ndarray, torch.Tensor or pd.DataFrame but got {type(array)}."
            )
        array = ak.values_astype(array, np.float32)
        array = ak.from_regular(array, axis=0)  # ensure regularity
        self.driver = AkwardJaggedAutoregressiveSequencer(
            features=array,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            pred_offset=pred_offset,
            padding_left_flag=padding_left_flag,
            warmup_steps=warmup_steps,
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
        pred_offset: int = 0,
        padding_left_flag: Union[bool, int] = True,
        warmup_steps: int = None,
    ):
        array = ak.values_astype(array, np.float32)
        ragged_dim = find_singular_ragged_dim(array)  # validate ragged dim

        if ragged_dim is None:
            # TODO: Proposed fix to ensure we always have 3dim output of self.driver.sequences_batch(idx) as one dim is added within.
            # if array.ndim == 2:
            #     array = ak.from_regular(array, axis=0)  # ensure regularity
            # elif array.ndim == 3:
            #     array = ak.from_regular(array, axis=1)  # ensure regularity
            # else:
            #     raise ValueError(
            #         "Input array do not have ragged dim and ndim is greater than 3. "
            #         "This case has not been tested."
            #     )
            array = ak.from_regular(array, axis=0)  # ensure regularity

            logger.warning(
                "Input array do not have ragged dim. Considering 0th axis as ragged."
            )

        self.driver = AkwardJaggedAutoregressiveSequencer(
            features=array,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            pred_offset=pred_offset,
            padding_left_flag=padding_left_flag,
            warmup_steps=warmup_steps,
        )

    def __len__(self):
        return len(self.driver)

    def get_index_array(self):
        return self.driver.get_index_array()

    def sequences_batch(self, idx):
        return self.driver.sequences_batch(idx)
