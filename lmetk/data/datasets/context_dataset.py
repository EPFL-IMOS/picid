from typing import List

import torch
from lightning_fabric.utilities.data import AttributeDict
from numpy import ndarray

from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import context_collate_fn
from lmetk.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset


class ContextBatchDataset(BaseDataset):
    def __init__(
        self,
        # context: torch.Tensor,
        # target: torch.Tensor,
        # time_features: torch.Tensor,
        data_dict: dict[str, ndarray | list[ndarray]],
        meta_data_dict: dict[str, any],
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int = 1,
        **kwargs,
    ):
        self.target_feature = torch.from_numpy(data_dict["target"])
        self.context_features = torch.from_numpy(data_dict["features"])
        self.timestamps = torch.from_numpy(data_dict["timestamps"])

        self.target_dataset = SlidingWindowBatchDataset(
            data_dict=dict(
                features=self.target_feature,
                timestamps=self.timestamps,
            ),
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )
        self.context_dataset = SlidingWindowBatchDataset(
            data_dict=dict(
                features=self.context_features,
                timestamps=self.timestamps,
            ),
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

        super().__init__(data_dict, **kwargs)

    def __len__(self):
        assert len(self.target_dataset) == len(
            self.context_dataset
        ), "Target and context sequencers must have the same length"
        return len(self.target_dataset)

    def __getitem__(self, batch_idx: List[int]):
        d = AttributeDict(
            {
                "batch_idx": batch_idx,
            }
        )
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.target_dataset.__getitem__(
            batch_idx
        )
        d["target"] = AttributeDict(
            {
                "x": seq_x,
                "y": seq_y,
                "x_mark": seq_x_mark,
                "y_mark": seq_y_mark,
            }
        )

        seq_x, seq_y, seq_x_mark, seq_y_mark = self.context_dataset.__getitem__(
            batch_idx
        )
        d["context"] = AttributeDict(
            {
                "x": seq_x,
                "y": seq_y,
                "x_mark": seq_x_mark,
                "y_mark": seq_y_mark,
            }
        )
        return AttributeDict(d)

    def get_collate_fn(self):
        return context_collate_fn
