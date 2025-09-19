import awkward as ak
import numpy as np
import torch
from typing import List, Optional, Tuple
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# AkwardJaggedAutoregressiveSequencer
# ----------------------------------------------------------------------
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
        # Determine which axis is ragged (same helper logic as AdaptiveTimeSeriesSequencer)
        from lmetk.transforms.base.multisource import find_singular_ragged_dim

        self.ragged_dim = find_singular_ragged_dim({"features": self.features})
        if self.ragged_dim is None:
            raise ValueError("No ragged dimension found in the input array.")

        # Count feature dimension
        example_index = (0,) * (self.ragged_dim + 1)
        example_element = self.features[example_index]
        if isinstance(example_element, (int, float, np.number)):
            self.num_features = 1
        else:
            self.num_features = len(example_element)

        # Precompute valid window bounds: (sample_idx..., start, end, r_start, p_end)
        self._indices = self._build_indices()

    def _build_indices(self) -> List[tuple]:
        from lmetk.utils.awkward_utils import ragged_index_tuples

        num_indexing_dims = self.ragged_dim + 1
        all_possible = ragged_index_tuples(self.features, num_indexing_dims)

        valid = []
        for index_tuple in all_possible:
            # index_tuple = (...sample dims..., t)
            sample_idx = index_tuple[: self.ragged_dim]
            t = index_tuple[self.ragged_dim]
            sample_len = len(self.features[sample_idx])
            # Compute required length for this start t
            required_end = (t + self.seq_len - self.label_len) + (
                self.label_len + self.pred_len
            )
            if required_end <= sample_len:
                # enforce stride: only keep starts that respect stride
                if (t % self.stride) == 0:
                    valid.append(index_tuple)
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
            seq_x_time : optional time features
            seq_y_time : optional time features
        """
        sample_idx = index_tuple[: self.ragged_dim]
        t = index_tuple[self.ragged_dim]
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

        if self.time_features is not None:
            tf_sample = self.time_features[sample_idx]
            seq_x_time = self._ak_to_tensor(tf_sample[x_start:x_end])
            seq_y_time = self._ak_to_tensor(tf_sample[y_start:y_end])
        else:
            seq_x_time = None
            seq_y_time = None

        return seq_x, seq_y, seq_x_time, seq_y_time


# ----------------------------------------------------------------------
# Simple Dataset wrapper
# ----------------------------------------------------------------------
class JaggedSequenceDataset(Dataset):
    def __init__(self, sequencer: AkwardJaggedAutoregressiveSequencer):
        self.seq = sequencer

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, idx: int):
        index_tuple = self.seq._indices[idx]
        return self.seq.get_window_for_index(index_tuple)


# features = ak.Array([[0, 1, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
features = ak_regularize_regular_axes(
    ak.Array(
        [
            [[0, 1], [2, 3]],  # Sample 0  # Sub-sample (0,0)
            [[10, 11]],  # Sample 1  # Sub-sample (1,0)
            [[4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
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

dataset = JaggedSequenceDataset(seq)

# The 3rd overall timestep in the data corresponds to index tuple (0, 1, 1)
for i in range(dataset.__len__()):
    seq_x, seq_y, _, _ = dataset[i]

    print("\n## Complex Data (N * var * C)")
    print(f"Detected ragged dimension: {sequencer_complex.ragged_dim}")
    print("seq_x:\n", seq_x)
    print("seq_y:\n", seq_y)
