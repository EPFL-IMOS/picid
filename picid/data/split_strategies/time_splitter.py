"""Time-based time-series splitter strategy."""

import logging
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from picid.data.split_strategies.base import TimeSeriesSplitter


class ValueWarning(Warning):
    pass


logger = logging.getLogger(__name__)


class TimeSplitter(TimeSeriesSplitter):
    def __init__(
        self,
        train: Optional[Union[int, float]] = None,
        val: Optional[Union[int, float]] = None,
        test: Optional[Union[int, float]] = None,
        seq_len: int = 384,
        label_len: int = 96,
        pred_len: int = 96,
        **kwargs,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.splits_dict = None
        self.create_splits_for = kwargs.get(
            "create_splits_for", ["features", "timestamps"]
        )

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

        elif all(
            isinstance(x, int)
            for x in [self.train, self.val, self.test]
            if x is not None
        ):
            train_len = self.train or 0
            val_len = self.val or 0
            test_len = self.test if self.test is not None else n - train_len - val_len

        else:
            raise ValueError("train/val/test must all be float or int (not mixed).")

        if self.test is None:
            assert (
                train_len + val_len + test_len == n
            ), "Train + val + test split sizes must sum to total data length."

        else:
            logger.info(f"Test size is specified and equal to: {self.test}")
            train_len + val_len + test_len
            assert (
                n >= (train_len + val_len + test_len)
            ), "The specified test size is not correct as the total num. of points n >= (train_len + val_len + test_len)"
            warnings.warn(
                f"Test size is specified and equal to: {self.test}, hence train_len + val_len + test_len = {train_len + val_len + test_len}, implying missing {n - (train_len + val_len + test_len)}",
                ValueWarning,
            )

        lookback = self.seq_len
        if lookback < 0:
            raise ValueError("pred_len must be <= seq_len")

        train_start = 0
        train_end = train_len

        val_start = max(0, train_end - lookback)
        val_end = val_start + val_len + lookback

        test_start = max(0, val_end - lookback)
        test_end = test_start + test_len + lookback

        used_samples = (
            (train_end - train_start)
            + (val_end - val_start - lookback)
            + (test_end - test_start - lookback)
        )

        if self.test is None:
            assert used_samples == n, (
                f"Not all data used: {used_samples} out of {n} samples. "
                "Ensure that train/val/test splits are set correctly."
            )
        else:
            assert all(
                isinstance(x, int)
                for x in [self.train, self.val, self.test]
                if x is not None
            ), f"Note that self.test is {self.test} and self.train, self.val, self.test should be int."

            assert used_samples == self.train + self.val + self.test, (
                f"Not all data used: {used_samples} out of {self.train + self.val + self.test} samples. "
                "Ensure that train/val/test splits are set correctly."
            )

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

        train_mask = np.ones(train_end - train_start, dtype=bool)
        val_mask = np.concatenate(
            [np.zeros(lookback, dtype=bool), np.ones(val_len, dtype=bool)]
        )
        test_mask = np.concatenate(
            [np.zeros(lookback, dtype=bool), np.ones(test_len, dtype=bool)]
        )

        masks = {
            "train": train_mask,
            "val": val_mask,
            "test": test_mask,
        }

        splits = {
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        }
        return splits, masks

    def split_data(
        self, data_dict: Dict[str, Any], split_variable: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split one payload into train, validation, and test slices.

        Parameters
        ----------
        data_dict : dict[str, Any]
            Dictionary containing the data.
        split_variable : str
            Name of the variable to split.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            Split payloads and the associated boolean masks.
        """
        data = data_dict[split_variable]

        if self.splits_dict is None:
            self.splits_dict, self.split_masks = self.get_splits(data)

        train_data = data[self.splits_dict["train"][0] : self.splits_dict["train"][1]]
        val_data = data[self.splits_dict["val"][0] : self.splits_dict["val"][1]]
        test_data = data[self.splits_dict["test"][0] : self.splits_dict["test"][1]]

        splitted_data = {"train": train_data, "val": val_data, "test": test_data}

        return splitted_data, self.split_masks

    def __repr__(self):
        return f"SimpleSplitter(train={self.train}, val={self.val}, test={self.test})"
