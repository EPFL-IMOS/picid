import numpy as np
from typing import Optional, Union, Tuple
from .base import TimeSeriesSplitter

class SimpleSplitter(TimeSeriesSplitter):
    def __init__(self,
                 train: Optional[Union[int, float]] = None,
                 val: Optional[Union[int, float]] = None,
                 test: Optional[Union[int, float]] = None,
                 seq_len: int = 384,
                 label_len: int = 96,
                 pred_len: int = 96):
        """
        Splits time series data into train, validation, and test sets.

        Args:
            train (int or float): Number of samples or proportion for training set.
            val (int or float): Number of samples or proportion for validation set.
            test (int or float): Number of samples or proportion for test set.

        """
        self.train = train
        self.val = val
        self.test = test

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.splits_dict = None

    def get_splits(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into train, validation, and test sets.
        Args:
            data (np.ndarray): The time series data to be split.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The train, validation, and test sets.
        """

        n = len(data)

        if all(isinstance(x, float) for x in [self.train, self.val, self.test] if x is not None):
            total_ratio = sum(x for x in [self.train, self.val, self.test] if x is not None)
            if total_ratio > 1.0:
                raise ValueError("Sum of proportions cannot exceed 1.0")
            train_len = int(n * (self.train or 0))
            val_len = int(n * (self.val or 0))
            test_len = n - train_len - val_len
        
        elif all(isinstance(x, int) for x in [self.train, self.val] if x is not None):
            train_len = self.train or 0
            val_len = self.val or 0
            test_len = self.test or 0
            if train_len + val_len + test_len > n:
                raise ValueError("Split sizes exceed data length.")
        else:
            raise ValueError("train/val/test must all be int or float (not mixed).")

        splits = {
            "train": (0, train_len),
            "val": (train_len, train_len + val_len),
            "test": (train_len + val_len, n)
        }
        return splits
    
    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the data into train, validation, and test sets.
        Args:
            data (np.ndarray): The time series data to be split.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The train, validation, and test sets.
        """
        
        # Check if self.splits_dict is None
        if self.splits_dict is None:
            self.splits_dict = self.get_splits(data)
        train_data = data[self.splits_dict["train"][0]:self.splits_dict["train"][1]]
        val_data = data[self.splits_dict["val"][0]:self.splits_dict["val"][1]]
        test_data = data[self.splits_dict["test"][0]:self.splits_dict["test"][1]]
        
        return train_data, val_data, test_data

    def __repr__(self):
        return f"SimpleSplitter(train={self.train}, val={self.val}, test={self.test})"
    