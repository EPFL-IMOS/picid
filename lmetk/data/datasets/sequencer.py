from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch

class IndexSequencer:
    
    def __init__(
        self, 
        length: int,  
        seq_len: int,   # window size
        label_len: int, # overlap size
        pred_len: int,  # prediction size
        stride: int) -> None:
        
        self.seq_len = seq_len
        self.label_len = label_len  
        self.pred_len = pred_len
        self.stride = stride
        self.device = torch.device("cpu")  # Force CPU indexing 
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
        
        
        max_start = length - self.seq_len - self.pred_len
        i = torch.arange(0, max_start + 1, self.stride)

        e = i + self.seq_len
        r = e - self.label_len
        p = r + self.label_len + self.pred_len

        assert torch.all(p <= length), "Some prediction ends exceed available data"

        self.i_e_bounds = torch.stack([i, e], dim=1).to(self.device)
        self.r_p_bounds = torch.stack([r, p], dim=1).to(self.device)

        
        
    def __len__(self):
        return len(self.i_e_bounds)
      
      
    def sequences_batch(self, index_subset: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor]:
        x_window = self.i_e_bounds[index_subset]
        x_mark_window = self.r_p_bounds[index_subset]

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
        stride: int) -> None:
        
        length = features.shape[0]
        super().__init__(length=length, seq_len=seq_len, label_len=label_len, pred_len=pred_len, stride=stride)
        
        self.features = torch.Tensor(features, device=self.device)
        self.time_features = torch.Tensor(time_features, device=self.device)
    
    
    
    def _create_sequences(self, x_window, x_mark_window):
        """
        Create sequences for the given windows.
        
        Input:
            x_window: List of tuples (start, end) for the input sequence
            x_mark_window: List of tuples (start, end) for the output sequence
        Output:
            seq_x: Encoder input sequence
            seq_y: Decoder input sequence
            seq_x_time: Time features for the encoder input sequence
            seq_y_time: Time features for the decoder output sequence
        """
        x_idx = torch.arange(x_window[0], x_window[1], device=self.device) # Shape: B
        y_idx = torch.arange(x_mark_window[0], x_mark_window[1], device=self.device) # Shape: B

        seq_x = self.features[x_idx]
        seq_y = self.time_features[y_idx]

        seq_x_time = self.features[x_idx]
        seq_y_time = self.features[y_idx]

        return seq_x, seq_y, seq_x_time, seq_y_time
