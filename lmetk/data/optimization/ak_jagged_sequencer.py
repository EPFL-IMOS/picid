# A simple logger/named transform input for the example
import logging
from typing import List, Optional, Tuple

import awkward as ak
import numpy as np
import torch

from lmetk.transforms.base.multisource import find_singular_ragged_dim
from lmetk.utils.awkward_utils import get_ak_shape, ragged_index_tuples

logger = logging.getLogger()


class AkwardJaggedAutoregressiveSequencer:
    """
    Autoregressive sequencer for *ragged* Awkward Arrays.

    Creates overlapping input/output pairs from a single ragged Awkward Array,
    analogous to AutoregressiveSequencer but operating on jagged segments.

    Currently we support (D = regular dim, var = ragged dim):
    D D
    D var D D
    var D
    var D D

    Args:
        features: Awkward Array of shape (N, variable_length, D) or deeper
        time_features: optional Awkward Array with the same ragged structure
        seq_len: Length of the encoder input window
        label_len: Overlap between encoder and decoder
        pred_len: Length of the prediction horizon
        stride: Step size between consecutive windows
        pad: Whether to pad sequences shorter than required (default False)
        pad_value: Padding value if pad=True
    """

    def __init__(
        self,
        features: ak.Array,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int = 1,
        padding_left_flag: int = 0,
        warmup_steps: int = None,
    ):
        self.features = features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride

        # Flags to use additional sequencer features
        # Partially (not implemented yet)
        self.padding_left_flag = padding_left_flag
        self.warmup_steps = warmup_steps

        self.device = torch.device("cpu")
        self._checked_output_shapes = False

        assert features.ndim < 4, "Features dim greater than 3 has not been tested"
        # Determine which axis is ragged
        self.ragged_dim = find_singular_ragged_dim({"features": self.features})
        self.is_ragged = True

        # Case when regular ndarray wrapped into awkward array is passed
        if self.ragged_dim is None:
            self.is_ragged = False

        # Get feature dimension
        self.num_features = get_ak_shape(features)[-1]

        # Precompute valid window bounds
        self._indices = np.array(self._build_indices())

    def _build_indices(self) -> List[tuple]:
        """
        Computation of valid (...prefix..., t) tuples.

        Key change:
            • iterate over parent prefixes only
            • compute valid t directly from the stored length
        """
        required_len = self.seq_len + self.pred_len
        stride = int(self.stride)

        # --- Path 1: Logic for DENSE arrays (is_ragged=False) ---
        # This block handles your Test Case 3 correctly.
        if not self.is_ragged:
            # For a dense array, we treat it as a single time series.
            total_len = len(self.features)
            if total_len < required_len:
                return []

            # Directly calculate the valid start times.
            starts = range(0, total_len - required_len + 1, self.stride)

            # For a single series, the sample index is always 0.
            # This returns a list of (sample_idx, time_idx) tuples, e.g., [(0, 0), (0, 2), ...].
            return [(t,) for t in starts]

        # 1. Get all parent prefixes (no t) for the ragged dimension
        prefixes = ragged_index_tuples(self.features, self.ragged_dim)

        # 2. Precompute segment lengths
        lengths = ak.to_list(ak.num(self.features, axis=self.ragged_dim))

        # 3. Helper to read length given a prefix tuple
        def get_length(prefix):
            if self.ragged_dim == 0:
                return lengths[prefix[0]] if prefix else lengths
            out = lengths
            for p in prefix:
                out = out[p]
            return out

        # 4. Build valid tuples without generating every candidate
        valid: List[tuple] = []
        for prefix in prefixes:
            seg_len = get_length(prefix)
            if seg_len >= required_len:
                # generate only valid starts respecting stride
                starts = range(0, seg_len - required_len + 1, stride)
                valid.extend([(*prefix, t) for t in starts])

        return valid

    def __len__(self) -> int:
        return len(self._indices)

    def get_index_array(self) -> np.ndarray:
        return self._indices

    def _get_window_for_index(
        self, batch_idx: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        idx_sel = self._indices[batch_idx]
        b_dim = idx_sel.shape[0]

        # Efficient way to create a touple of 1d arrays for selection
        # (arr_index_0, arr_index_1, arr_index_1) = tuple(idx_sel.T)
        idx_sel_t = tuple(idx_sel.T)

        # In this case, we have no (ragged) trunk data, only one long time series
        # so we do need to select in the first dimension
        select_in_d0 = len(idx_sel_t) == 1

        # Now point is defined as the coordinate where the sequence starts
        # in the last dimension of the tuple
        now_point = idx_sel_t[-1]
        x_range = np.arange(0, self.seq_len)[None, :].repeat(b_dim, axis=0)
        y_range = np.arange(0, self.label_len + self.pred_len)[None, :].repeat(
            b_dim, axis=0
        )

        x_range = ak.Array(x_range + now_point[:, None])
        y_range = ak.Array(y_range + now_point[:, None] + self.seq_len - self.label_len)

        # if there is no trunk, we need to select in the
        # first dimension and introduce the batch dimension
        if select_in_d0:
            # In this case we want to introduce a new batch dimension, therefore,
            # we keep all dimensions regular (see next comment below)
            leaf_data = self.features
        else:
            sel_dims = idx_sel_t[:-1]
            leaf_data = self.features[sel_dims]
            # ak_from regular transforms the regular dim -1 into a ragged dim (of the selection index).
            # This way, it selects properly the segments in the var dim of trunk data
            # if the dim remains singular, a new dimension is created
            # e.g. (var, 1) -> (B, var, 1)
            x_range = ak.from_regular(x_range, axis=-1)
            y_range = ak.from_regular(y_range, axis=-1)

        seq_x = torch.from_numpy(leaf_data[x_range].to_numpy())
        seq_y = torch.from_numpy(leaf_data[y_range].to_numpy())

        # Ensure the output shapes are correct (check it only once), n dim is correct as well, shape is (T,C) meaning timestamps, features
        if not self._checked_output_shapes:
            assert (
                seq_x.shape[1] == self.seq_len
            ), f"seq_x length mismatch: expected {self.seq_len}, got {seq_x.shape[1]}"
            assert (
                seq_y.shape[1] == self.label_len + self.pred_len
            ), f"seq_y length mismatch: expected {self.label_len + self.pred_len}, got {seq_y.shape[0]}"
            assert (
                seq_x.shape[2] == self.num_features
            ), f"seq_x feature dim mismatch: expected {self.num_features}, got {seq_x.shape[2]}"
            assert (
                seq_y.shape[2] == self.num_features
            ), f"seq_y feature dim mismatch: expected {self.num_features}, got {seq_y.shape[2]}"
            self._checked_output_shapes = True

        return seq_x, seq_y

    def sequences_batch(
        self, batch_idx: List[int]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Generate a batch of sequences for the given indices.

        Args:
            batch_idx: A list of integer indices corresponding to the sequences to generate.

        Returns:
            A tuple of tensors containing the batched sequences:
            (seq_x_batch, seq_y_batch)
        """
        return self._get_window_for_index(batch_idx)
