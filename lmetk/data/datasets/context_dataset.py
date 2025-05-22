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
        stride: int = 24) -> None:
        
        self.seq_len = seq_len
        self.label_len = label_len  
        self.pred_len = pred_len
        self.stride = stride
        
        assert stride >= 1, "stride must be greater than or equal to 1"
        assert seq_len >= label_len, "seq_len must be greater than or equal to label_len"
        assert label_len + pred_len <= seq_len, "label_len + pred_len must be less than or equal to seq_len"
        assert length >= seq_len + pred_len, "Not enough data points for even a single sample"
        # Question: can stride be smaller than prediction length? In such case during the validation/test we predict the same variable multiple times with different context
        assert stride <= pred_len, "stride must be less than or equal to pred_len"

        
        # !!!!!
        # TODO R.: Check what is this for ;) 
        # if dt_ts_out is not None:
        #     dt_ts_out.sort()
        #     dt_ts_out = np.array(dt_ts_out)
        #     dt_max = dt_ts_out[-1]
        #     upper_limit = max(ts_out, dt_max + 1)
        # else:
        #     upper_limit = ts_out
        
        
        # # That is basically analog to how we get samples in classical seqntial dataset
        # # s = index
        # # e = index + self.seq_len
        # i = torch.arange(0, length - seq_len - 1, stride)
        # self.start_end_index = torch.stack([i, i + seq_len], dim=1)
        
        # # r = e - self.label_len
        # # p = r + self.label_len + self.pred_len
        # # seq_len - label_len
        # i = torch.arange(stride, length - seq_len, stride)
        # self.second = torch.stack([i, i + self.label_len + self.pred_len], dim=1)
        
        max_start = length - self.seq_len - self.pred_len
        i = torch.arange(0, max_start + 1, self.stride)

        e = i + self.seq_len
        r = e - self.label_len
        p = r + self.label_len + self.pred_len

        assert torch.all(p <= length), "Some prediction ends exceed available data"

        self.start_end_index = torch.stack([i, e], dim=1)
        self.second = torch.stack([r, p], dim=1)

        
        
    def __len__(self):
        return len(self.start_end_index)
      
      
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
        time_features: torch.Tensor,
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        stride: int=1) -> None:
        
        length = features.shape[0]
        super().__init__(length=length, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
        
        self.features = features
        self.time_features = time_features 
    
    # This acts as batch collate function
    
    def _create_sequences(self, x_window, x_mark_window):
        x_idx = [torch.arange(start, end) for start, end in [x_window]]
        y_idx = [torch.arange(start, end) for start, end in [x_mark_window]]

        x_idx = torch.stack(x_idx)  # Shape: [B, seq_len]
        y_idx = torch.stack(y_idx)  # Shape: [B, label_len + pred_len]

        seq_x = self.features[x_idx]
        seq_y = self.time_features[y_idx]

        seq_x_time = self.features[x_idx]
        seq_y_time = self.features[y_idx]

        return seq_x, seq_y, seq_x_time, seq_y_time

    # def _create_sequences(self, x_window: List[Tuple[int]], x_mark_window: List[Tuple[int]]) -> Tuple[torch.Tensor]:
        
    #     seq_x = self.features[x_window, ...]
    #     seq_y = self.time_features[x_mark_window, ...]

    #     seq_x_time = self.features[x_window, ...]
    #     seq_y_time = self.features[x_mark_window, ...]
        
    #     return seq_x, seq_y, seq_x_time, seq_y_time

        
        

class ContextBatchDataset(torch.utils.data.Dataset):
    def __init__(self, 
        
        features: torch.Tensor,
        time_features: torch.Tensor,
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
        
        self.target_sequencer = AutoregressiveSequencer(features=target_feature, time_features=time_features, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
        self.context_sequencer = AutoregressiveSequencer(features=context_features, time_features=time_features, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
                    
    def __len__(self):
        assert len(self.target_sequencer) == len(self.context_sequencer), "Target and context sequencers must have the same length"
        return len(self.target_sequencer)

    def __getitem__(self, batch_idx: List[int]): 
        d = {
            "batch_idx": batch_idx,
        }
        seq_x, seq_y, seq_x_time, seq_y_time = self.target_sequencer.sequences_batch(batch_idx)
        d['target'] = {
            "x": seq_x,
            "y": seq_y,
            "x_time": seq_x_time,
            "y_time": seq_y_time
        }

        
        seq_x, seq_y, seq_x_time, seq_y_time = self.context_sequencer.sequences_batch(batch_idx)
        d['context'] = {
            "x": seq_x,
            "y": seq_y,
            "x_time": seq_x_time,
            "y_time": seq_y_time
        }
        return d
