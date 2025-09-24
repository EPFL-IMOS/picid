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

from lmetk.utils.awkward_utils import ak_regularize_regular_axes, get_ak_shape

from lmetk.transforms.base.multisource import find_singular_ragged_dim
from lmetk.utils.awkward_utils import ragged_index_tuples


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
        pad: bool = False,
        pad_value: float = 0.0,
    ):
        self.features = features
        self.time_features = time_features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.pad = pad
        self.pad_value = pad_value
        self.device = torch.device("cpu")

        assert features.ndim < 4, "Features dim greater than 3 has not been tested"
        # Determine which axis is ragged
        self.ragged_dim = find_singular_ragged_dim({"features": self.features})
        self.is_ragged = True

        # Case when regular ndarray wrapped into awkward array is passed
        if self.ragged_dim is None:
            # # This is the case
            # assert (
            #     get_ak_shape(self.features)[-1] == 1
            # ), "Ensuring we are working with targets"
            self.is_ragged = False

        # Count feature dimension
        example_index = (0,) * (self.ragged_dim or 0 + 1)
        example_element = self.features[example_index]
        if isinstance(example_element, (int, float, np.number)):
            self.num_features = 1
        else:
            self.num_features = len(example_element)

        # Precompute valid window bounds
        self._indices = self._build_indices_efficient()  # TODO: old self._build_indices

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

    def _pad_and_fill(self, array: ak.Array, target_len: int) -> ak.Array:
        if len(array) == 0:
            shape = (target_len, self.num_features)
            return ak.from_numpy(np.full(shape, self.pad_value, dtype=np.float32))

        padded = ak.pad_none(array, target_len, axis=0)
        builder = ak.ArrayBuilder()
        pad_record = np.full(self.num_features, self.pad_value, dtype=np.float32)
        for record in padded:
            if record is None:
                builder.append(pad_record)
            elif self.num_features == 1 and isinstance(record, (int, float, np.number)):
                builder.append(np.array([record], dtype=np.float32))
            else:
                builder.append(record)
        return builder.snapshot()

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

        if self.pad:
            seq_x_ak = self._pad_and_fill(seq_x_ak, self.seq_len)
            seq_y_ak = self._pad_and_fill(seq_y_ak, self.label_len + self.pred_len)

        seq_x = self._ak_to_tensor(seq_x_ak)
        seq_y = self._ak_to_tensor(seq_y_ak)

        # Ensure the output shapes are correct (check it only once), n dim is correct as well
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
        if isinstance(batch_idx, int):
            batch_idx = [batch_idx]

        batch_items = []
        for i in batch_idx:
            # index_tuple
            index_tuple = self._indices[i]
            item = self.get_window_for_index(index_tuple)
            batch_items.append(item)

        # Separate the components of the batch using zip

        # Move from list of tuple to the tuple of lists
        # e.g. [(x1, y1), (x2, y2), (x3, y3)] -> (list[x1, x2, x3], list[y1, y2, y3])
        seq_x_list, seq_y_list = zip(*batch_items)

        # Stack the individual tensors to create the batch dimension: list[(seq_len, C),(seq_len, C),...] -> Tensor[B, seq_len, C]
        seq_x_batch = torch.stack(seq_x_list, dim=0)
        seq_y_batch = torch.stack(seq_y_list, dim=0)

        return seq_x_batch, seq_y_batch


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
