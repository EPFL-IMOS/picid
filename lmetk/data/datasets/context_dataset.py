from typing import List, Dict
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
        # Make sure that the data_dict contains the required keys
        required_keys = ["features", "time_features"]
        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"Data dictionary must contain the key: '{key}'")

        self.target_feature = data_dict["target"]
        self.context_features = data_dict["features"]
        self.time_features = data_dict["time_features"]

        self.target_dataset = SlidingWindowBatchDataset(
            data_dict=dict(
                features=self.target_feature,
                time_features=self.time_features,
            ),
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )
        self.context_dataset = SlidingWindowBatchDataset(
            data_dict=dict(
                features=self.context_features,
                time_features=self.time_features,
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
        target: Dict = self.target_dataset.__getitem__(batch_idx)
        d["target"] = AttributeDict(target)

        context: Dict = self.context_dataset.__getitem__(batch_idx)
        d["context"] = AttributeDict({context})
        return AttributeDict(d)

    def get_collate_fn(self):
        return context_collate_fn
