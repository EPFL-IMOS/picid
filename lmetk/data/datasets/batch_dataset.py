from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
from lmetk.data.datasets.sequencer import AutoregressiveSequencer
class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, 
        
        features: torch.Tensor,
        time_features: torch.Tensor,
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        stride: int = 1,
        ):
            
        self.feature_sequencer = AutoregressiveSequencer(
            features=features,
            time_features=time_features, 
            seq_len=seq_len, 
            label_len=label_len, 
            pred_len=pred_len, 
            stride=stride
            )
        
    def __len__(self):
        return self.feature_sequencer.__len__() # len(self.feature_sequencer)

    def __getitem__(self, batch_idx: List[int]): 
        seq_x, seq_y, seq_x_time, seq_y_time = self.feature_sequencer.sequences_batch(batch_idx)
        return seq_x, seq_y, seq_x_time, seq_y_time