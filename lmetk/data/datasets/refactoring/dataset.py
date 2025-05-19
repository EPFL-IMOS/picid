"""Refactored dataset class for time series data."""

from torch.utils.data import Dataset
import numpy as np
class TimeSeriesDataset(Dataset):
    """Standard 'sliding windon' Time Series Dataset Class."""
    def __init__(self, 
                 data: np.ndarray,
                 time_features: np.ndarray,
                 seq_len: int,
                 label_len: int,
                 pred_len: int):
        self.data = data
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        s = index
        e = s + self.seq_len
        r = e - self.label_len
        p = r + self.label_len + self.pred_len

        seq_x = self.data[s:e]
        seq_y = self.data[r:p]
        seq_x_mark = self.time_features[s:e]
        seq_y_mark = self.time_features[r:p]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
