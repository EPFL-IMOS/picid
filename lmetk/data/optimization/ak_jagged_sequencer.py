import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional

# A simple logger/named transform input for the example
import logging

logger = logging.getLogger()

from lmetk.utils.awkward_utils import (
    ak_find_var_dims,
    ak_regularize_regular_axes,
    ragged_index_tuples,
)

from lmetk.transforms.base.multisource import find_singular_ragged_dim


class AdaptiveTimeSeriesSequencer:
    """
    Creates time-series windows from a single ragged Awkward Array.
    """

    def __init__(
        self,
        features: ak.Array,
        seq_len: int = 12,
        label_len: int = 6,
        pred_len: int = 1,
        pad: bool = True,
        pad_value: float = 0.0,
    ):

        self.features = features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.pad = pad
        self.pad_value = pad_value

        self.ragged_dim = find_singular_ragged_dim({"features": self.features})
        if self.ragged_dim is None:
            raise ValueError("No ragged dimension found in the input array.")

        example_index = (0,) * (self.ragged_dim + 1)
        example_element = self.features[example_index]
        if isinstance(example_element, (int, float, np.number)):
            self.num_features = 1
        else:
            self.num_features = len(example_element)

        self._indices = self._build_indices()

    def _build_indices(self) -> list[tuple]:
        """Builds a master list of indices, filtering for complete windows if padding is off."""
        num_indexing_dims = self.ragged_dim + 1
        all_possible_indices = ragged_index_tuples(self.features, num_indexing_dims)
        if self.pad:
            return all_possible_indices

        valid_indices = []
        for index_tuple in all_possible_indices:
            sample_idx = index_tuple[: self.ragged_dim]
            t = index_tuple[self.ragged_dim]
            sample_len = len(self.features[sample_idx])
            required_len = (t + self.seq_len - self.label_len) + (
                self.label_len + self.pred_len
            )
            if required_len <= sample_len:
                valid_indices.append(index_tuple)
        return valid_indices

    def __len__(self) -> int:
        return len(self._indices)

    def _pad_and_fill(self, array: ak.Array, target_len: int) -> ak.Array:
        """Pads a sliced array to the target length."""
        if len(array) == 0:
            shape = (target_len, self.num_features)
            return ak.from_numpy(np.full(shape, self.pad_value, dtype=np.float32))

        padded = ak.pad_none(array, target_len, axis=0)
        builder = ak.ArrayBuilder()
        pad_record = np.full(self.num_features, self.pad_value, dtype=np.float32)
        for record in padded:
            # Here, we must ensure scalars are appended in a way that matches the pad_record structure
            if record is None:
                builder.append(pad_record)
            elif self.num_features == 1 and isinstance(record, (int, float, np.number)):
                builder.append(np.array([record], dtype=np.float32))  # Wrap scalar
            else:
                builder.append(record)
        return builder.snapshot()

    def _ak_to_tensor(self, array: ak.Array) -> torch.Tensor:
        """Converts a finalized Awkward array to a PyTorch tensor."""
        np_arr = ak.to_numpy(array)
        if self.num_features == 1 and np_arr.ndim == 1:
            np_arr = np_arr.reshape(-1, 1)
        return torch.from_numpy(np_arr).to(torch.float32)

    def get_window_for_index(
        self, index_tuple: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts, pads (if enabled), and converts a single window."""
        sample_idx = index_tuple[: self.ragged_dim]
        t = index_tuple[self.ragged_dim]
        sample = self.features[sample_idx]

        x_start, x_end = t, t + self.seq_len
        y_start = t + self.seq_len - self.label_len
        y_end = y_start + self.label_len + self.pred_len

        seq_x_ak = sample[x_start:x_end]
        seq_y_ak = sample[y_start:y_end]

        # --- KEY FIX IS HERE ---
        # Only call the padding function when self.pad is True.
        if self.pad:
            seq_x_ak = self._pad_and_fill(seq_x_ak, self.seq_len)
            seq_y_ak = self._pad_and_fill(seq_y_ak, self.label_len + self.pred_len)

        return self._ak_to_tensor(seq_x_ak), self._ak_to_tensor(seq_y_ak)


class SequenceDataset(Dataset):
    """Standard PyTorch Dataset wrapper."""

    def __init__(self, sequencer: AdaptiveTimeSeriesSequencer):
        self.seq = sequencer

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index_tuple = self.seq._indices[idx]
        return self.seq.get_window_for_index(index_tuple)


if __name__ == "__main__":

    features_complex = ak.Array([[0, 1, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
    print("PADDING is set to FALSE")
    # The sequencer automatically detects ragged_dim = 2
    sequencer_complex = AdaptiveTimeSeriesSequencer(
        features=features_complex,
        seq_len=2,
        label_len=1,
        pred_len=1,
        pad=False,
        pad_value=-1,
    )

    dataset_complex = SequenceDataset(sequencer_complex)
    # The 3rd overall timestep in the data corresponds to index tuple (0, 1, 1)
    for i in range(dataset_complex.__len__()):
        seq_x, seq_y = dataset_complex[i]

        print("\n## Complex Data (N * var )")
        print(f"Detected ragged dimension: {sequencer_complex.ragged_dim}")
        print("seq_x:\n", seq_x)
        print("seq_y:\n", seq_y)

    print("PADDING is set to TRUE")

    # The sequencer automatically detects ragged_dim = 2
    sequencer_complex = AdaptiveTimeSeriesSequencer(
        features=features_complex,
        seq_len=2,
        label_len=1,
        pred_len=1,
        pad=True,
        pad_value=-1,
    )

    dataset_complex = SequenceDataset(sequencer_complex)
    # The 3rd overall timestep in the data corresponds to index tuple (0, 1, 1)
    for i in range(dataset_complex.__len__()):
        seq_x, seq_y = dataset_complex[i]

        print("\n## Complex Data (N * var )")
        print(f"Detected ragged dimension: {sequencer_complex.ragged_dim}")
        print("seq_x:\n", seq_x)
        print("seq_y:\n", seq_y)

    features_complex = ak_regularize_regular_axes(
        ak.Array(
            [
                [[0, 1], [2, 3]],  # Sample 0  # Sub-sample (0,0)
                [[10, 11]],  # Sample 1  # Sub-sample (1,0)
                [[4, 5], [6, 7], [8, 9]],
            ]
        )
    )
    print("PADDING is set to FALSE")
    # The sequencer automatically detects ragged_dim = 2
    sequencer_complex = AdaptiveTimeSeriesSequencer(
        features=features_complex,
        seq_len=2,
        label_len=1,
        pred_len=1,
        pad=False,
        pad_value=-1,
    )

    dataset_complex = SequenceDataset(sequencer_complex)
    # The 3rd overall timestep in the data corresponds to index tuple (0, 1, 1)
    for i in range(dataset_complex.__len__()):
        seq_x, seq_y = dataset_complex[i]

        print("\n## Complex Data (N * var * C)")
        print(f"Detected ragged dimension: {sequencer_complex.ragged_dim}")
        print("seq_x:\n", seq_x)
        print("seq_y:\n", seq_y)

    print("PADDING is set to TRUE")

    # The sequencer automatically detects ragged_dim = 2
    sequencer_complex = AdaptiveTimeSeriesSequencer(
        features=features_complex,
        seq_len=2,
        label_len=1,
        pred_len=1,
        pad=True,
        pad_value=-1,
    )

    dataset_complex = SequenceDataset(sequencer_complex)
    # The 3rd overall timestep in the data corresponds to index tuple (0, 1, 1)
    for i in range(dataset_complex.__len__()):
        seq_x, seq_y = dataset_complex[i]

        print("\n## Complex Data (N * var * C)")
        print(f"Detected ragged dimension: {sequencer_complex.ragged_dim}")
        print("seq_x:\n", seq_x)
        print("seq_y:\n", seq_y)
