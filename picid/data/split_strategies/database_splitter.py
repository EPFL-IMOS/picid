"""
Timestamp-based time-series splitter strategy for forecasting datasets.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from picid.data.split_strategies.base import TimeSeriesSplitter


class TimeStampSplitter(TimeSeriesSplitter):
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
        **kwargs,
    ):
        """
        Create a timestamp splitter with optional lookback handling.

        Parameters
        ----------
        test_start : str
            Timestamp when the test split starts.
        test_end : Optional[str]
            Optional end of the test period. If None, goes to end of data.
        train_ratio : float
            Proportion of pre-test data for training.
        val_ratio : float
            Proportion of pre-test data for validation.
        seq_len : int
            Length of input sequence for forecasting.
        label_len : int
            Not used in splitting, passed for consistency.
        pred_len : int
            Prediction length (used in sanity checks).
        frequency : str
            Frequency for aligning split ends, e.g., 'D' for day.
        apply_lookback : bool
            Whether to shift val/test start index back by seq_len.
        **kwargs : Any
            Compatibility keyword arguments.
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
        self.masks = None
        self.create_splits_for = kwargs.get(
            "create_splits_for", ["features", "timestamps"]
        )

        if not np.isclose(train_ratio + val_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio must equal 1.0")

    def _round_to_period_end(self, dates: pd.Series, index: int) -> int:
        """
        Snap an index to the next full period boundary.

        Parameters
        ----------
        dates : pandas.Series
            Timestamp series used for boundary alignment.
        index : int
            Position to round forward.

        Returns
        -------
        int
            Search index for the next period boundary.
        """
        ts = dates.iloc[index]
        rounded = ts.floor(self.frequency) + pd.tseries.frequencies.to_offset(
            self.frequency
        )
        if rounded > dates.iloc[-1]:
            return len(dates)
        return dates.searchsorted(rounded)

    def get_splits(self, data: Any, dates: Any) -> dict:
        if not isinstance(dates, pd.Series):
            raise ValueError("`dates` must be a pandas Series.")

        if not isinstance(dates.iloc[0], pd.Timestamp):
            try:
                dates = pd.to_datetime(dates)
            except Exception:
                raise ValueError("`dates` must be a pandas Series of timestamps.")

        if len(dates) != len(data):
            raise ValueError("Length of `dates` must match length of `data`.")

        n = len(data)
        lookback = self.seq_len
        min_required = lookback + self.pred_len

        i_test_start = dates.searchsorted(self.test_start)
        i_test_end = dates.searchsorted(self.test_end) if self.test_end else n

        n_pre_test = i_test_start
        i_train_end_est = int(self.train_ratio * n_pre_test)
        i_train_end = self._round_to_period_end(dates, i_train_end_est)

        i_val_start = i_train_end
        i_val_end = self._round_to_period_end(
            dates, i_val_start + int(self.val_ratio * n_pre_test)
        )

        if self.apply_lookback:
            i_val_start = max(0, i_val_start - lookback)
            i_test_start = max(0, i_test_start - lookback)
        else:
            lookback = 0

        assert (i_train_end - 0) >= min_required, "Train split too short."
        assert (i_val_end - i_val_start) >= min_required, "Validation split too short."
        assert (i_test_end - i_test_start) >= min_required, "Test split too short."

        train_mask = np.ones(i_train_end - 0, dtype=bool)
        val_mask = np.concatenate(
            [
                np.zeros(lookback, dtype=bool),
                np.ones(i_val_end - (i_val_start + lookback), dtype=bool),
            ]
        )
        test_mask = np.concatenate(
            [
                np.zeros(lookback, dtype=bool),
                np.ones(i_test_end - (i_test_start + lookback), dtype=bool),
            ]
        )

        masks = {
            "train": train_mask,
            "val": val_mask,
            "test": test_mask,
        }

        splits_dict = {
            "train": (0, i_train_end),
            "val": (i_val_start, i_val_end),
            "test": (i_test_start, i_test_end),
        }
        return splits_dict, masks

    def split_data(
        self, data_dict: Dict[str, Any], split_variable: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split one payload into train, validation, and test slices.

        Parameters
        ----------
        data_dict : Dict[str, Any]
            Dictionary containing the data to split.
        split_variable : str
            Name of the variable to split.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            Split payloads and the associated boolean masks.
        """
        data = data_dict[split_variable]
        timestamps = data_dict["timestamps"]
        if self.splits_dict is None:
            self.splits_dict, self.masks = self.get_splits(data, timestamps)

        train = data[self.splits_dict["train"][0] : self.splits_dict["train"][1]]
        val = data[self.splits_dict["val"][0] : self.splits_dict["val"][1]]
        test = data[self.splits_dict["test"][0] : self.splits_dict["test"][1]]

        splitted_data = {"train": train, "val": val, "test": test}

        return splitted_data, self.masks

    def __repr__(self):
        return (
            f"TimeStampSplitter(test_start={self.test_start}, test_end={self.test_end}, "
            f"train_ratio={self.train_ratio}, val_ratio={self.val_ratio}, "
            f"seq_len={self.seq_len}, pred_len={self.pred_len}, "
            f"apply_lookback={self.apply_lookback}, freq='{self.frequency}')"
        )
