from typing import List

import torch
from lmetk.data.optimization.sequencer import AutoregressiveSequencer


class SWBatchDataset(torch.utils.data.Dataset):
    """Efficient implementation of Standard 'sliding window' Time Series Dataset Class."""
    def __init__(
        self,
        features: torch.Tensor,
        time_features: torch.Tensor,
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int,
        **kwargs
    ):
        # Make sure that the stide covers all the data

        
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
