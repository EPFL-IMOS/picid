"""Refactored dataset class for time series features."""

from torch.utils.data import Dataset
import numpy as np


class SWDataset(Dataset):
    """Standard 'sliding window' Time Series Dataset Class."""

    def __init__(
        self,
        features: np.ndarray,
        time_features: np.ndarray,
        seq_len: int,
        label_len: int,
        pred_len: int,
    ):
        # Type checking
        assert isinstance(features, np.ndarray), "features must be a numpy array"
        assert isinstance(time_features, np.ndarray), (
            "time_features must be a numpy array"
        )
        assert isinstance(seq_len, int) and seq_len > 0, (
            "seq_len must be a positive integer"
        )
        assert isinstance(label_len, int) and label_len > 0, (
            "label_len must be a positive integer"
        )
        assert isinstance(pred_len, int) and pred_len > 0, (
            "pred_len must be a positive integer"
        )
        # Dimensional validation
        assert len(features) == len(time_features), (
            "features and time_features must have the same length"
        )
        assert seq_len >= label_len, (
            "seq_len must be greater than or equal to label_len"
        )
        assert len(features) >= seq_len + pred_len, (
            "Not enough features points for even a single sample"
        )

        self.features = features
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        # Validate index is within bounds
        # assert 0 <= index < len(self), f"Index {index} out of range [0, {len(self)})"

        s = index
        e = s + self.seq_len
        r = e - self.label_len
        p = r + self.label_len + self.pred_len

        # Ensure slicing indices are valid
        assert 0 <= s < e <= len(self.features), "Input sequence indices out of bounds"
        assert 0 <= r < p <= len(self.features), "Target sequence indices out of bounds"

        seq_x = self.features[s:e]
        seq_y = self.features[r:p]
        seq_x_mark = self.time_features[s:e]
        seq_y_mark = self.time_features[r:p]

        # Verify output shapes
        assert len(seq_x) == self.seq_len, "Input sequence length mismatch"
        assert len(seq_y) == self.label_len + self.pred_len, (
            "Target sequence length mismatch"
        )
        assert len(seq_x_mark) == self.seq_len, "Input time features length mismatch"
        assert len(seq_y_mark) == self.label_len + self.pred_len, (
            "Target time features length mismatch"
        )

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # Ensure indexing remains within bounds for full prediction window
        # self.seq_len because we added self.seq_len
        return len(self.features) - self.seq_len - self.pred_len + 1
