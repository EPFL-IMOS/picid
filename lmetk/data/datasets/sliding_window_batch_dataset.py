from typing import Dict, List, Optional, Union

import awkward as ak
import numpy as np
import torch

from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import collate_key_value_batch
from lmetk.data.optimization.sequencer import (
    BaseSequencer,
    DenseArraySequencer,
    RaggedArraySequencer,
)

ArrayLike = Union[np.ndarray, torch.Tensor, ak.Array]


class SlidingWindowBatchDataset(BaseDataset):
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
        suffixes=["_seq_x", "_seq_y"],
        **kwargs,
    ):

        self.sequencers: Dict[str, BaseSequencer] = {}
        self._length: Optional[int] = None
        self.suffixes = suffixes

        for key, arr in data_dict.items():
            if arr is None:
                continue

            if isinstance(arr, (np.ndarray, torch.Tensor)):
                driver = DenseArraySequencer(
                    array=arr,
                    seq_len=seq_len,
                    label_len=label_len,
                    pred_len=pred_len,
                    stride=stride,
                )

            elif isinstance(arr, ak.Array):
                driver = RaggedArraySequencer(
                    array=arr,
                    seq_len=seq_len,
                    label_len=label_len,
                    pred_len=pred_len,
                    stride=stride,
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
        Returns a dict of {key: (x, y)}.
        For drivers without time features the last two entries are None.
        """
        out = {}
        for k, drv in self.sequencers.items():
            x, y = drv.sequences_batch(idx)
            # seq_x: (B, seq_len, C),
            # seq_y: (B, label_len + pred_len, C);
            # note that y can be empty tensor; (B,T,C)
            out[f"{k}{self.suffixes[0]}"] = x
            out[f"{k}{self.suffixes[1]}"] = y
        return out

    def get_collate_fn(self):
        return collate_key_value_batch
