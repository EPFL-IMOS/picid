from typing import List, Dict
from lightning_fabric.utilities.data import AttributeDict
import torch

from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.context_dataset import ContextBatchDataset
from lmetk.data.datasets.collate_functions import context_collate_fn
from lmetk.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset

import awkward as ak

# @override decorator
from typing import override


class RULContextBatchDataset(ContextBatchDataset):
    """
    Dataset for RUL prediction with context features.
    Combines target RUL sequences with context feature sequences.
    """

    @override
    def __getitem__(self, batch_idx: List[int]):
        d = AttributeDict(
            {
                "batch_idx": batch_idx,
            }
        )
        target: Dict = self.target_dataset.__getitem__(batch_idx)
        seq_x, _ = target["rul"]

        target["rul"] = seq_x[
            :, -1
        ]  # (B=1, seq_len, C) -> (B=1, C), take the last time step, collate fn will concatenate over the batch dim

        d["target"] = AttributeDict(target)

        context: Dict = self.context_dataset.__getitem__(batch_idx)
        d["context"] = AttributeDict(context)
        return AttributeDict(d)

    def get_collate_fn(self):
        return context_collate_fn
