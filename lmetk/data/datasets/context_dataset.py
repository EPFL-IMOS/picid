from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
from lmetk.data.datasets.batch_dataset import BatchDataset

class ContextBatchDataset(torch.utils.data.Dataset):
    def __init__(self, 
        
        features: torch.Tensor,
        time_features: torch.Tensor,
        mask: torch.Tensor, # True for target, False for the rest of covariates
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        stride: int = 1,
        ):
        
        target_feature = features[..., mask]
        context_features = features[..., ~mask]
        
        self.target_dataset = BatchDataset(features=target_feature, time_features=time_features, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
        self.context_dataset = BatchDataset(features=context_features, time_features=time_features, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
                    
    def __len__(self):
        assert len(self.target_dataset) == len(self.context_dataset), "Target and context sequencers must have the same length"
        return len(self.target_dataset)

    def __getitem__(self, batch_idx: List[int]): 
        d = {
            "batch_idx": batch_idx,
        }
        seq_x, seq_y, seq_x_time, seq_y_time = self.target_dataset.__getitem__(batch_idx)
        d['target'] = {
            "x": seq_x,
            "y": seq_y,
            "x_time": seq_x_time,
            "y_time": seq_y_time
        }

        
        seq_x, seq_y, seq_x_time, seq_y_time = self.target_dataset.__getitem__(batch_idx)
        d['context'] = {
            "x": seq_x,
            "y": seq_y,
            "x_time": seq_x_time,
            "y_time": seq_y_time
        }
        return d


# class ContextBatchDataset(torch.utils.data.Dataset):
#     def __init__(self, 
        
#         features: torch.Tensor,
#         time_features: torch.Tensor,
#         mask: torch.Tensor, # True for target, False for the rest of covariates
#         seq_len: int,   # window size
#         label_len: int, # overlap size
#         pred_len: int,  # prediction size
#         stride: int = 1,
#         ):
        
#         target_feature = features[..., mask]
#         context_features = features[..., ~mask]
        
#         self.target_sequencer = AutoregressiveSequencer(features=target_feature, time_features=time_features, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
#         self.context_sequencer = AutoregressiveSequencer(features=context_features, time_features=time_features, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
                    
#     def __len__(self):
#         assert len(self.target_sequencer) == len(self.context_sequencer), "Target and context sequencers must have the same length"
#         return len(self.target_sequencer)

#     def __getitem__(self, batch_idx: List[int]): 
#         d = {
#             "batch_idx": batch_idx,
#         }
#         seq_x, seq_y, seq_x_time, seq_y_time = self.target_sequencer.sequences_batch(batch_idx)
#         d['target'] = {
#             "x": seq_x,
#             "y": seq_y,
#             "x_time": seq_x_time,
#             "y_time": seq_y_time
#         }

        
#         seq_x, seq_y, seq_x_time, seq_y_time = self.context_sequencer.sequences_batch(batch_idx)
#         d['context'] = {
#             "x": seq_x,
#             "y": seq_y,
#             "x_time": seq_x_time,
#             "y_time": seq_y_time
#         }
#         return d
