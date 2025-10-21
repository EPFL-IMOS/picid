from typing import List, Dict
from lightning_fabric.utilities.data import AttributeDict
from numpy import ndarray

from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import (
    collate_key_value_batch,
)
from lmetk.data.datasets.sliding_window_batch_dataset import SlidingWindowBatchDataset

import awkward as ak


class ContextBatchDataset(BaseDataset):
    def __init__(
        self,
        data_dict: dict[str, ndarray | list[ndarray]],
        task_type: str,
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        pred_offset: int = 0,
        stride: int = 1,
        padding_left_flag: int = 0,
        warmup_steps: int = None,
        **kwargs,
    ):

        # Make sure that the data_dict contains the required keys
        if task_type in ["rul", "ahrul", "soc"]:
            required_keys = ["features", "rul"]  # "time_features", "target"
            context_key = "features"
            target_key = task_type

        elif task_type in ["forecasting"]:
            required_keys = ["features", "target"]  # "time_features", "target"
            context_key = "features"
            target_key = "target"
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"Data dictionary must contain the key: '{key}'")

        self.target_feature = data_dict[target_key]
        self.context_features = data_dict[context_key]

        # We batch two separate sequences in parallel (one for context, one for target)
        # They must have the same length (number of samples)
        dataset_shared_params = dict(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            padding_left_flag=padding_left_flag,
            warmup_steps=warmup_steps,
            pred_offset=pred_offset,
        )

        self.target_dataset = SlidingWindowBatchDataset(
            data_dict={target_key: self.target_feature}, **dataset_shared_params
        )
        self.context_dataset = SlidingWindowBatchDataset(
            data_dict={context_key: self.context_features}, **dataset_shared_params
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
        target: Dict = self.target_dataset.__getitem__(batch_idx)
        d["target"] = AttributeDict(target)

        context: Dict = self.context_dataset.__getitem__(batch_idx)
        d["context"] = AttributeDict(context)
        return AttributeDict(d)

    def get_collate_fn(self):
        return collate_key_value_batch
