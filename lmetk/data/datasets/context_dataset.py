from typing import List

import torch
from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import context_collate_fn
from lmetk.data.datasets.sw_batch_dataset import SWBatchDataset
from lightning_fabric.utilities.data import AttributeDict

class ContextBatchDataset(BaseDataset):
    def __init__(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        time_features: torch.Tensor,
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int = 1,
        **kwargs
    ):
        self.target_feature = target
        self.context_features = context
        self.time_features = time_features

        self.target_dataset = SWBatchDataset(
            features=self.target_feature,
            time_features=self.time_features,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )
        self.context_dataset = SWBatchDataset(
            features=self.context_features,
            time_features=self.time_features,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

    def __len__(self):
        assert len(self.target_dataset) == len(self.context_dataset), (
            "Target and context sequencers must have the same length"
        )
        return len(self.target_dataset)

    def __getitem__(self, batch_idx: List[int]):
        d = AttributeDict({
            "batch_idx": batch_idx,
        })
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.target_dataset.__getitem__(
            batch_idx
        )
        d["target"] = AttributeDict({
            "x": seq_x,
            "y": seq_y,
            "x_mark": seq_x_mark,
            "y_mark": seq_y_mark,
        })

        seq_x, seq_y, seq_x_mark, seq_y_mark = self.context_dataset.__getitem__(
            batch_idx
        )
        d["context"] = AttributeDict({
            "x": seq_x,
            "y": seq_y,
            "x_mark": seq_x_mark,
            "y_mark": seq_y_mark,
        })
        return AttributeDict(d)
    
    @staticmethod
    def collate_fn(batch):
        return context_collate_fn(batch)
