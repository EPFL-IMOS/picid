import numpy as np
from typing import Optional, Union, Tuple
from .base import TimeSeriesSplitter


class SimpleSplitter(TimeSeriesSplitter):
    def __init__(
        self,
        train: Optional[Union[int, float]] = None,
        val: Optional[Union[int, float]] = None,
        test: Optional[Union[int, float]] = None,
        seq_len: int = 384,
        label_len: int = 96,
        pred_len: int = 96,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.splits_dict = None

    def get_splits(self, data: np.ndarray) -> dict:
        n = len(data)

        if all(
            isinstance(x, float)
            for x in [self.train, self.val, self.test]
            if x is not None
        ):
            total_ratio = sum(
                x for x in [self.train, self.val, self.test] if x is not None
            )
            if total_ratio > 1.0:
                raise ValueError("Sum of proportions cannot exceed 1.0")

            train_len = int(n * (self.train or 0))
            val_len = int(n * (self.val or 0))
            test_len = n - train_len - val_len

        elif all(isinstance(x, int) for x in [self.train, self.val] if x is not None):
            train_len = self.train or 0
            val_len = self.val or 0
            test_len = self.test if self.test is not None else n - train_len - val_len

            if train_len + val_len + test_len != n:
                raise ValueError(
                    "Train + val + test split sizes must sum to total data length."
                )
        else:
            raise ValueError("train/val/test must all be float or int (not mixed).")

        # Compute overlap between segments
        lookback = self.seq_len  # - self.pred_len
        if lookback < 0:
            raise ValueError("pred_len must be <= seq_len")

        # Split ranges with overlap
        train_start = 0
        train_end = train_len

        val_start = max(0, train_end - lookback)
        val_end = val_start + val_len + lookback

        test_start = max(0, val_end - lookback)
        test_end = test_start + test_len + lookback

        # Final safety check: all data must be used
        used_samples = (
            (train_end - train_start)
            + (val_end - val_start - lookback)
            + (test_end - test_start - lookback)
        )
        assert used_samples == n, (
            f"Not all data used: {used_samples} out of {n} samples. "
            "Ensure that train/val/test splits are set correctly."
        )

        # Each split must be long enough for one sample
        min_required = self.seq_len + self.pred_len
        assert (train_end - train_start) >= min_required, (
            f"Train split too short ({train_end - train_start} samples) for seq_len={self.seq_len} "
            f"and pred_len={self.pred_len}. Need at least {min_required}."
        )
        assert (val_end - val_start) >= min_required, (
            f"Validation split too short ({val_end - val_start} samples) for seq_len={self.seq_len} "
            f"and pred_len={self.pred_len}. Need at least {min_required}."
        )
        assert (test_end - test_start) >= min_required, (
            f"Test split too short ({test_end - test_start} samples) for seq_len={self.seq_len} "
            f"and pred_len={self.pred_len}. Need at least {min_required}."
        )

        return {
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        }

    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.splits_dict is None:
            self.splits_dict = self.get_splits(data)
        train_data = data[self.splits_dict["train"][0] : self.splits_dict["train"][1]]
        val_data = data[self.splits_dict["val"][0] : self.splits_dict["val"][1]]
        test_data = data[self.splits_dict["test"][0] : self.splits_dict["test"][1]]
        return train_data, val_data, test_data

    def __repr__(self):
        return f"SimpleSplitter(train={self.train}, val={self.val}, test={self.test})"
