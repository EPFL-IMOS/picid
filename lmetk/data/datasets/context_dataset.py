from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

class IndexSequencer:
    
    def __init__(
        self, # F: torch.Tensor,
        length: int,  
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        dt_ts_out: List[int] = None,
        stride: int = 1) -> None:
        
        self.seq_len = seq_len
        self.label_len = label_len  
        self.pred_len = pred_len
        self.stride = stride
        # !!!!!
        # TODO R.: Check what is this for ;) 
        # if dt_ts_out is not None:
        #     dt_ts_out.sort()
        #     dt_ts_out = np.array(dt_ts_out)
        #     dt_max = dt_ts_out[-1]
        #     upper_limit = max(ts_out, dt_max + 1)
        # else:
        #     upper_limit = ts_out
        
        
        # OLD CODE
        # i = torch.arange(0, length - (ts_in + upper_limit) + 1, stride=stride)
        # self.start_end_index = torch.stack ([i, i + (ts_in + upper_limit)], dim=1)
        
        # --Questinable update--
        i = torch.arange(0, length - seq_len - pred_len + 1, stride=stride)
        
        # s = index
        # e = index + self.seq_len
        self.start_end_index = torch.stack([i, i + seq_len], dim=1)
        
        # r = e - self.label_len
        # p = r + self.label_len + self.pred_len
        self.second = torch.stack([(i[1:] - label_len), (i[1:] + label_len + pred_len)], dim=1)
        
        self.N_sequences = len(self.start_end_index)
        
    def __len__(self):
        return self.N_sequences
      
      
    def sequences_batch(self, index_subset: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor]:
        x_window = self.start_end_index[index_subset]
        x_mark_window = self.second[index_subset]

        return self._create_sequences(x_window, x_mark_window)
    
    @abstractmethod
    def _create_sequences(self, x_window: List[Tuple[int]], x_mark_window: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        pass

   
class AutoregressiveSequencer(IndexSequencer):
    
    def __init__(
        self, 
        features: torch.Tensor,
        timestamps: torch.Tensor,
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        stride: int=10) -> None:
        
        length = features.shape[0]
        super().__init__(length=length, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
        
        self.features = features
        self.timestamps = timestamps 
    
    # This acts as batch collate function
    def _create_sequences(self, x_window: List[Tuple[int]], x_mark_window: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        
        seq_x = self.features[x_window, ...]
        seq_y = self.timestamps[x_mark_window, ...]

        seq_x_time = self.features[x_window, ...]
        seq_y_time = self.features[x_mark_window, ...]
        
        return seq_x, seq_y, seq_x_time, seq_y_time

        # F, T = self.F, self.T
        
        # f_unfolded, t_unfolded = [], []
        # for i, j in sel:
        #     f_unfolded.append(F[i : i + ts_in, ...])
        #     t_unfolded.append(T[i + ts_in : j, ...])
        
        # f_unfolded = torch.stack(f_unfolded, dim=0)
        # t_unfolded = torch.stack(t_unfolded, dim=0)
        
        # if self.in_features_mask is not None:
        #     f_unfolded = f_unfolded[..., self.in_features_mask]
            
        # if self.out_features_mask is not None:
        #     t_unfolded = t_unfolded[..., self.out_features_mask]
        

class ContextBatchDataset(torch.utils.data.Dataset):
    def __init__(self, 
        
        features: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor, # True for target, False for the rest of covariates
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        stride: int = 1,
        #Features: torch.Tensor, 
        #Context: torch.Tensor,
        #ts_in: int,
        #ts_out:int,
        #out_features_mask: Optional[torch.Tensor] = None,
        
        ):

        target_feature = features[..., mask]
        context_features = features[..., ~mask]
        
        self.target = AutoregressiveSequencer(features=target_feature, timestamps=timestamps, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
        self.context = AutoregressiveSequencer(features=context_features, timestamps=timestamps, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
                    
    def __len__(self):
        return len(self.target)

    def __getitem__(self, batch_idx: List[int]): 
        d = {
            "batch_idx": batch_idx,
        }
        seq_x, seq_y, seq_x_time, seq_y_time = self.target(batch_idx)
        d['target'] = {
            "x": seq_x,
            "y": seq_y,
            "x_time": seq_x_time,
            "y_time": seq_y_time
        }

        
        seq_x, seq_y, seq_x_time, seq_y_time = self.context(batch_idx)
        d['context'] = {
            "x": seq_x,
            "y": seq_y,
            "x_time": seq_x_time,
            "y_time": seq_y_time
        }
        return d
