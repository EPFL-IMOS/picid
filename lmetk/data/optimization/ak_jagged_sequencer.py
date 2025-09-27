import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

# A simple logger/named transform input for the example
import logging

logger = logging.getLogger()

from lmetk.transforms.base.multisource import find_singular_ragged_dim
from lmetk.utils.awkward_utils import (
    ragged_index_tuples,
    ak_regularize_regular_axes,
    get_ak_shape,
)


class AkwardJaggedAutoregressiveSequencer:
    """
    Autoregressive sequencer for *ragged* Awkward Arrays.

    Creates overlapping input/output pairs from a single ragged Awkward Array,
    analogous to AutoregressiveSequencer but operating on jagged segments.

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
        time_features: Optional[ak.Array],
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int = 1,
    ):
        self.features = features
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.device = torch.device("cpu")

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
        self._indices = np.array(self._build_indices_efficient())

    def _build_indices(self) -> List[tuple]:
        num_indexing_dims = self.ragged_dim + 1 if self.is_ragged else 1
        all_possible = ragged_index_tuples(self.features, num_indexing_dims)

        # TODO: huge room for improvement here, vactorize operations
        valid = []
        for index_tuple in all_possible:
            # index_tuple = (...sample dims..., t)
            sample_idx = index_tuple[: self.ragged_dim or 0]
            t = index_tuple[self.ragged_dim or 0]
            sample_len = len(self.features[sample_idx])
            # The total length required for one sequence starting at t
            required_len = self.seq_len + self.pred_len
            if t + required_len <= sample_len:
                # Enforce stride: only keep starts that respect the stride
                if (t % self.stride) == 0:
                    valid.append(index_tuple)
        return valid

    def _build_indices_efficient(self) -> List[tuple]:
        """
        Much faster computation of valid (...prefix..., t) tuples.

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

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _ak_to_tensor(self, array: ak.Array) -> torch.Tensor:
        np_arr = ak.to_numpy(array)
        if self.num_features == 1 and np_arr.ndim == 1:
            np_arr = np_arr.reshape(-1, 1)
        return torch.from_numpy(np_arr).to(torch.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_window_for_index(
        self, index_tuple: tuple
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Extract a single encoder/decoder pair.

        Returns:
            seq_x : (seq_len, D) tensor
            seq_y : (label_len + pred_len, D) tensor
        """
        sample_idx = index_tuple[: self.ragged_dim or 0]
        t = index_tuple[self.ragged_dim or 0]
        sample = self.features[sample_idx]

        x_start, x_end = t, t + self.seq_len
        y_start = t + self.seq_len - self.label_len
        y_end = y_start + self.label_len + self.pred_len

        seq_x_ak = sample[x_start:x_end]
        seq_y_ak = sample[y_start:y_end]

        seq_x = self._ak_to_tensor(seq_x_ak)
        seq_y = self._ak_to_tensor(seq_y_ak)

        # Ensure the output shapes are correct (check it only once), n dim is correct as well, shape is (T,C) meaning timestamps, features
        if not hasattr(self, "_checked_output_shapes"):
            assert (
                seq_x.shape[0] == self.seq_len
            ), f"seq_x length mismatch: expected {self.seq_len}, got {seq_x.shape[0]}"
            assert (
                seq_y.shape[0] == self.label_len + self.pred_len
            ), f"seq_y length mismatch: expected {self.label_len + self.pred_len}, got {seq_y.shape[0]}"
            assert (
                seq_x.shape[1] == self.num_features
            ), f"seq_x feature dim mismatch: expected {self.num_features}, got {seq_x.shape[1]}"
            assert (
                seq_y.shape[1] == self.num_features
            ), f"seq_y feature dim mismatch: expected {self.num_features}, got {seq_y.shape[1]}"
            self._checked_output_shapes = True

        return seq_x, seq_y

    def get_window_for_index_efficient(self, batch_idx):
        idx_sel = self._indices[batch_idx]
        b_dim = idx_sel.shape[0]

        idx_sel_t = tuple(idx_sel.T)

        # In this case, we have no trunk data, only one long time series
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
            trunk_data = ak.values_astype(self.features, np.float32)
        else:
            sel_dims = idx_sel_t[:-1]
            trunk_data = ak.values_astype(self.features[sel_dims], np.float32)
            # ak_from regular transforms the regular dim -1 into a ragged dim (of the selection index).
            # This way, it selects properly the segments in the var dim of trunk data
            # if the dim remains singular, a new dimension is created
            # e.g. (var, 1) -> (B, var, 1)
            x_range = ak.from_regular(x_range, axis=-1)
            y_range = ak.from_regular(y_range, axis=-1)

        seq_x_batch = torch.from_numpy(trunk_data[x_range].to_numpy())
        seq_y_batch = torch.from_numpy(trunk_data[y_range].to_numpy())

        return seq_x_batch, seq_y_batch

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
        return self.get_window_for_index_efficient(batch_idx)
        # if isinstance(batch_idx, int):
        #     batch_idx = [batch_idx]

        # batch_items = []
        # for i in batch_idx:
        #     # index_tuple
        #     index_tuple = self._indices[i]
        #     item = self.get_window_for_index(index_tuple)
        #     batch_items.append(item)

        # # Separate the components of the batch using zip

        # # Move from list of tuple to the tuple of lists
        # # e.g. [(x1, y1), (x2, y2), (x3, y3)] -> (list[x1, x2, x3], list[y1, y2, y3])
        # seq_x_list, seq_y_list = zip(*batch_items)

        # # Stack the individual tensors to create the batch dimension: list[(seq_len, C),(seq_len, C),...] -> Tensor[B, seq_len, C]
        # seq_x_batch = torch.stack(seq_x_list, dim=0)
        # seq_y_batch = torch.stack(seq_y_list, dim=0)

        # return seq_x_batch, seq_y_batch


if __name__ == "__main__":
    print("### --- Test Case 1: Simple Ragged Data --- ###")
    features = ak.Array([[0, 1, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    time_feats = ak.Array([[0], [1], [2]])

    seq = AkwardJaggedAutoregressiveSequencer(
        features=features,
        time_features=time_feats,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=2,
    )

    for i in range(seq.__len__()):
        seq_x, seq_y, seq_x_mark, seq_y_mark = seq.sequences_batch([i])

        print("\n## Complex Data (N * var)")
        print("seq_x:\n", seq_x)
        print("seq_x_mark:\n", seq_x_mark)
        print("seq_y:\n", seq_y)
        print("seq_y_mark:\n", seq_y_mark)

    print("========================================")
    print("### --- Test Case 2: Three dim Ragged Data --- ###")
    features = ak_regularize_regular_axes(
        ak.Array(
            [
                [[0, 1], [2, 3]],  # Sample 0  # Sub-sample (0,0)
                [[10, 11]],  # Sample 1  # Sub-sample (1,0)
                [[4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
            ]
        )
    )
    time_feats = ak.Array([[0], [1], [2]])

    seq = AkwardJaggedAutoregressiveSequencer(
        features=features,
        time_features=time_feats,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=2,
    )

    for i in range(seq.__len__()):
        seq_x, seq_y, seq_x_mark, seq_y_mark = seq.sequences_batch([i])

        print("\n## Complex Data (N * 1)")
        print("seq_x:\n", seq_x)
        print("seq_x_mark:\n", seq_x_mark)
        print("seq_y:\n", seq_y)
        print("seq_y_mark:\n", seq_y_mark)

    print("========================================")

    print("### --- Test Case 3: No ragged dim --- ###")
    features = ak_regularize_regular_axes(ak.Array([[0], [1], [3], [4], [5], [6], [7]]))
    time_feats = None

    seq = AkwardJaggedAutoregressiveSequencer(
        features=features,
        time_features=time_feats,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=2,
    )

    for i in range(seq.__len__()):
        seq_x, seq_y, seq_x_mark, seq_y_mark = seq.sequences_batch([i])

        print("\n## Complex Data (N * var)")
        print("seq_x:\n", seq_x)
        print("seq_x_mark:\n", seq_x_mark)
        print("seq_y:\n", seq_y)
        print("seq_y_mark:\n", seq_y_mark)

    print("========================================")

    print("### --- Test Case 4: No ragged dim --- ###")
    features = ak_regularize_regular_axes(
        ak.Array(
            [
                [0, 2, 3],
                [1, 3, 4],
                [2, 4, 5],
                [3, 5, 6],
                [4, 6, 7],
                [5, 7, 8],
                [6, 8, 9],
            ]
        )
    )
    time_feats = None

    seq = AkwardJaggedAutoregressiveSequencer(
        features=features,
        time_features=time_feats,
        seq_len=2,
        label_len=0,
        pred_len=1,
        stride=2,
    )

    for i in range(seq.__len__()):
        seq_x, seq_y, seq_x_mark, seq_y_mark = seq.sequences_batch([i])

        print("\n## Complex Data (N * var)")
        print("seq_x:\n", seq_x)
        print("seq_x_mark:\n", seq_x_mark)
        print("seq_y:\n", seq_y)
        print("seq_y_mark:\n", seq_y_mark)
