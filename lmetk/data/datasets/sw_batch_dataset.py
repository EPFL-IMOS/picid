from typing import List

from numpy import ndarray
import torch
from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import forecasting_collate_fn
from lmetk.data.optimization.sequencer import AutoregressiveSequencer


class SWBatchDataset(BaseDataset):
    """Efficient implementation of Standard 'sliding window' Time Series Dataset Class."""
    def __init__(
        self,
        # features: torch.Tensor,
        # time_features: torch.Tensor,
        data_dict: dict[str, ndarray | list[ndarray]],         
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int,
        **kwargs
    ):  
        # Make sure that the data_dict contains the required keys
        required_keys = ["features", "time_features"]
        for key in required_keys:
            if key not in data_dict:
                raise ValueError(f"Data dictionary must contain the key: '{key}'")
            
        # Convert the data_dict["features"] and data_dict["time_features"] to torch.Tensor
        features = torch.Tensor(data_dict["features"])
        time_features = torch.Tensor(data_dict["time_features"])

        # TODO: Make sure that the stride covers all the data
        self.feature_sequencer = AutoregressiveSequencer(
            features=features,
            time_features=time_features,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

    def __len__(self):
        return self.feature_sequencer.__len__()  # len(self.feature_sequencer)

    def __getitem__(self, batch_idx: List[int]):
        """
        Get a batch of sequences based on the provided indices.
        Args:
            batch_idx (List[int]): Indices of the sequences to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - seq_x: Input sequences.
                - seq_y: Target sequences.
                - seq_x_time: Time features for input sequences.
                - seq_y_time: Time features for target sequences.
        """
        seq_x, seq_y, seq_x_time, seq_y_time = self.feature_sequencer.sequences_batch(
            batch_idx
        )
        return seq_x, seq_y, seq_x_time, seq_y_time

    def collate_fn(self):
        return forecasting_collate_fn