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


# ----------------------------------------------------------------------
# Unified dataset
# ----------------------------------------------------------------------
class SlidingWindowBatchDataset(Dataset):
    """
    Create a separate sequencer for every key in `data_dict`.
    Time features are NOT required.
    """

    def __init__(
        self,
        data_dict: Dict[str, ArrayLike],
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
        pad: bool = False,
        pad_value: float = 0.0,
    ):

        self.sequencers: Dict[str, BaseSequencer] = {}
        self._length: Optional[int] = None

        for key, arr in data_dict.items():
            if arr is None:
                continue

            if isinstance(arr, (np.ndarray, torch.Tensor)):
                driver = DenseDriver(arr, seq_len, label_len, pred_len, stride)
            elif isinstance(arr, ak.Array):
                driver = RaggedDriver(
                    arr, seq_len, label_len, pred_len, stride, pad, pad_value
                )
            else:
                raise NotImplementedError(
                    f"Unsupported type for key '{key}': {type(arr)}"
                )

            self.sequencers[key] = driver

            # ensure all sequencers have the same length
            if self._length is None:
                self._length = len(driver)
            elif len(driver) != self._length:
                raise ValueError(
                    f"All sequences must yield the same number of windows "
                    f"(mismatch at key '{key}')"
                )

    def __len__(self):
        return self._length or 0

    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, tuple]:
        """
        Returns a dict of {key: (x, y, x_time, y_time)}.
        For drivers without time features the last two entries are None.
        """
        out = {}
        for k, drv in self.sequencers.items():
            x, y, x_time, y_time = drv.sequences_batch(idx)
            out[k] = (x, y, x_time, y_time)
        return out

    def get_collate_fn(self):
        from lmetk.data.datasets.collate_functions import forecasting_collate_fn

        return forecasting_collate_fn
