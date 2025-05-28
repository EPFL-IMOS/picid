import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .base import TimeSeriesSplitter


class DateBasedSplitter(TimeSeriesSplitter):
    def __init__(
        self,
        test_start: str,
        test_end: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.3,
        seq_len: int = 384,
        label_len: int = 96,
        pred_len: int = 96,
        frequency: str = "D",
        apply_lookback: bool = False,
    ):
        """
        Parameters:
            test_start (str): Timestamp when the test split starts.
            test_end (Optional[str]): Optional end of the test period. If None, goes to end of data.
            train_ratio (float): Proportion of pre-test data for training.
            val_ratio (float): Proportion of pre-test data for validation.
            seq_len (int): Length of input sequence for forecasting.
            label_len (int): Not used in splitting, passed for consistency.
            pred_len (int): Prediction length (used in sanity checks).
            frequency (str): Frequency for aligning split ends, e.g., 'D' for day.
            apply_lookback (bool): Whether to shift val/test start index back by seq_len.
        """
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end) if test_end else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.frequency = frequency
        self.apply_lookback = apply_lookback
        self.splits_dict = None

        if not np.isclose(train_ratio + val_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio must equal 1.0")

    def _round_to_period_end(self, dates: pd.Series, index: int) -> int:
        """Snap index to the next full period boundary (e.g., next day)."""
        ts = dates.iloc[index]
        rounded = ts.floor(self.frequency) + pd.tseries.frequencies.to_offset(
            self.frequency
        )
        if rounded > dates.iloc[-1]:
            return len(dates)
        return dates.searchsorted(rounded)

    def get_splits(self, data: np.ndarray, dates: pd.Series) -> dict:
        if not isinstance(dates, pd.Series) or not isinstance(
            dates.iloc[0], pd.Timestamp
        ):
            raise ValueError("`dates` must be a pandas Series of timestamps.")
        if len(dates) != len(data):
            raise ValueError("Length of `dates` must match length of `data`.")

        n = len(data)
        lookback = self.seq_len
        min_required = lookback + self.pred_len

        i_test_start = dates.searchsorted(self.test_start)
        i_test_end = dates.searchsorted(self.test_end) if self.test_end else n

        # Pre-test portion to split into train/val
        n_pre_test = i_test_start
        i_train_end_est = int(self.train_ratio * n_pre_test)
        i_train_end = self._round_to_period_end(dates, i_train_end_est)

        i_val_start = i_train_end
        i_val_end = self._round_to_period_end(
            dates, i_val_start + int(self.val_ratio * n_pre_test)
        )

        # Apply lookback adjustments
        if self.apply_lookback:
            i_val_start = max(0, i_val_start - lookback)
            i_test_start = max(0, i_test_start - lookback)

        # Sanity checks
        assert (i_train_end - 0) >= min_required, "Train split too short."
        assert (i_val_end - i_val_start) >= min_required, "Validation split too short."
        assert (i_test_end - i_test_start) >= min_required, "Test split too short."

        self.splits_dict = {
            "train": (0, i_train_end),
            "val": (i_val_start, i_val_end),
            "test": (i_test_start, i_test_end),
        }
        return self.splits_dict

    def split_data(
        self, data: np.ndarray, dates: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.splits_dict is None:
            self.get_splits(data, dates)
        train = data[self.splits_dict["train"][0] : self.splits_dict["train"][1]]
        val = data[self.splits_dict["val"][0] : self.splits_dict["val"][1]]
        test = data[self.splits_dict["test"][0] : self.splits_dict["test"][1]]
        return train, val, test

    def __repr__(self):
        return (
            f"DateBasedSplitter(test_start={self.test_start}, test_end={self.test_end}, "
            f"train_ratio={self.train_ratio}, val_ratio={self.val_ratio}, "
            f"seq_len={self.seq_len}, pred_len={self.pred_len}, "
            f"apply_lookback={self.apply_lookback}, freq='{self.frequency}')"
        )
