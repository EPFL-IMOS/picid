from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch


class IndexSequencer:
    """
    Base class for creating sequences from time series data with configurable windowing.

    This class handles the indexing logic for creating sliding windows over time series data,
    supporting overlapping sequences and prediction horizons. It's designed to be subclassed
    for specific sequencing strategies.

    Args:
        length: Total number of data points in the time series
        seq_len: Length of the input sequence window
        label_len: Length of the overlap between input and output sequences
        pred_len: Length of the prediction horizon
        stride: Step size between consecutive windows

    Raises:
        AssertionError: If parameters don't satisfy the required constraints
    """

    def __init__(
        self,
        length: int,
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int,
    ) -> None:

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.device = torch.device("cpu")  # Force CPU indexing
        assert stride >= 1, "stride must be greater than or equal to 1"
        assert (
            seq_len >= label_len
        ), "seq_len must be greater than or equal to label_len"
        # assert label_len + pred_len <= seq_len, (
        #     "label_len + pred_len must be less than or equal to seq_len"
        # )
        assert (
            length >= seq_len + pred_len
        ), f"Not enough data points for even a single sample, length {length} < seq_len: {seq_len} + pred_len: {pred_len}"
        # Question: can stride be smaller than prediction length? In such case during the validation/test we predict the same variable multiple times with different context
        # assert stride <= pred_len, "stride must be less than or equal to pred_len"

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
        """Return the number of sequences that can be generated."""
        return len(self.i_e_bounds)

    def sequences_batch(
        self, index_subset: torch.Tensor | np.ndarray
    ) -> Tuple[torch.Tensor]:
        """
        Generate a batch of sequences for the given indices.

        Args:
            index_subset: Indices of the sequences to generate

        Returns:
            Tuple of tensors containing the generated sequences
        """
        x_window = self.i_e_bounds[index_subset]
        y_window = self.r_p_bounds[index_subset]

        return self._create_sequences(x_window, y_window)

    @abstractmethod
    def _create_sequences(
        self, x_window: List[Tuple[int]], y_window: List[Tuple[int]]
    ) -> Tuple[torch.Tensor]:
        """
        Create sequences from the given window boundaries.

        This method must be implemented by subclasses to define how sequences
        are extracted from the data.

        Args:
            x_window: List of tuples (start, end) for the input sequence windows
            x_mark_window: List of tuples (start, end) for the output sequence windows

        Returns:
            Tuple of tensors containing the sequences
        """
        pass


class AutoregressiveSequencer(IndexSequencer):
    """
    Concrete implementation of IndexSequencer for autoregressive sequence generation.

    This sequencer creates overlapping input-output pairs from time series data,
    suitable for training autoregressive models. It handles both feature data
    and time-based features.

    Args:
        features: Time series feature data of shape (time_steps, n_features)
        time_features: Time-based features of shape (time_steps, n_time_features)
        seq_len: Length of the input sequence window
        label_len: Length of the overlap between input and output sequences
        pred_len: Length of the prediction horizon
        stride: Step size between consecutive windows
    """

    def __init__(
        self,
        features: torch.Tensor,
        time_features: Optional[torch.Tensor],
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int,
    ) -> None:
        length = features.shape[0]
        super().__init__(
            length=length,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

        self.features = (
            features.to(self.device) if features.device != self.device else features
        )

        if time_features is not None:
            self.time_features = (
                time_features.to(self.device)
                if time_features.device != self.device
                else time_features
            )
        else:
            self.time_features = None

    def _create_sequences(self, x_window, y_window):
        """
        Create sequences for the given windows.

        Args:
            x_window: Tensor of shape (batch_size, 2) with (start, end) indices for input sequences
            x_mark_window: Tensor of shape (batch_size, 2) with (start, end) indices for output sequences

        Returns:
            tuple: A tuple containing:
                - seq_x: Encoder input sequence of shape (batch_size, seq_len, n_features)
                - seq_y: Decoder input sequence of shape (batch_size, label_len + pred_len, n_features)
                - seq_x_time: Time features for encoder input of shape (batch_size, seq_len, n_time_features)
                - seq_y_time: Time features for decoder output of shape (batch_size, label_len + pred_len, n_time_features)
        """
        x_idx = torch.arange(x_window[0], x_window[1], device=self.device)  # Shape: B
        y_idx = torch.arange(y_window[0], y_window[1], device=self.device)  # Shape: B

        seq_x = self.features[x_idx]
        seq_y = self.features[y_idx]

        seq_x_time = (
            self.time_features[x_idx] if self.time_features is not None else None
        )
        seq_y_time = (
            self.time_features[y_idx] if self.time_features is not None else None
        )

        return seq_x, seq_y, seq_x_time, seq_y_time


# TODO: we do not use it anywhere
class TrivialSequencer(IndexSequencer):
    """
    This trivial sequencer is not autoregressive, but simply returns the present time point t.
    It is used to have aligned features when used in combination with AutoregressiveSequencer

    Args:
        features: Time series feature data of shape (time_steps, n_features)
        seq_len: Length of the input sequence window
        label_len: Length of the overlap between input and output sequences
        pred_len: Length of the prediction horizon
        stride: Step size between consecutive windows
    """

    def __init__(
        self,
        features: torch.Tensor,
        seq_len: int,  # window size
        label_len: int,  # overlap size
        pred_len: int,  # prediction size
        stride: int,
    ) -> None:
        length = features.shape[0]
        self.features = features

        super().__init__(
            length=length,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
        )

    def _create_sequences(self, x_window, y_window):
        """
        Create sequences for the given windows.

        Args:
            x_window: Tensor of shape (batch_size, 2) with (start, end) indices for input sequences
            x_mark_window: Tensor of shape (batch_size, 2) with (start, end) indices for output sequences

        Returns:
            tuple: A tuple containing:
                - seq_x: Encoder input sequence of shape (batch_size, seq_len, n_features)
                - seq_y: Decoder input sequence of shape (batch_size, label_len + pred_len, n_features)
                - seq_x_time: Time features for encoder input of shape (batch_size, seq_len, n_time_features)
                - seq_y_time: Time features for decoder output of shape (batch_size, label_len + pred_len, n_time_features)
        """
        indicies = x_window[1] - 1
        seq_x = self.features[indicies]
        return seq_x
