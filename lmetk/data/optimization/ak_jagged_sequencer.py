import logging
from typing import List, Optional, Tuple, Union

import awkward as ak
import numpy as np
import torch

from lmetk.transforms.base.multisource import find_singular_ragged_dim
from lmetk.utils.awkward_utils import get_ak_shape, ragged_index_tuples

logger = logging.getLogger(__name__)


class AkwardJaggedAutoregressiveSequencer:
    """
    Autoregressive sequencer for *ragged* (variable-length) Awkward Arrays.

    This class generates overlapping input/output sliding windows from time-series data.
    It is designed to handle **Multi-Unit (Jagged)** data where each unit (e.g., an engine,
    a battery cycle) has a different temporal length. It can also handle **Dense**
    (single time-series) arrays.

    Key Features:
    -------------
    1.  **Jagged Support**: Iterates over the first dimension (Units) and slices time
        windows relative to the start of each unit.
    2.  **Strict Windowing (Default)**: Ensures input sequences consist entirely of
        real data. Short units are dropped.
    3.  **Left Padding (Warmup)**: Allows predictions to start earlier in the time
        series by padding the input history with zeros.

    Sequence Structure:
    -------------------
    For a given time step ``t`` (the anchor point):
    - **Input (seq_x)**:  ``[t - seq_len : t]``
    - **Target (seq_y)**: ``[t + pred_offset : t + pred_offset + label_len + pred_len]``

    Args:
        features (ak.Array): Input data array.
            - Shape (Ragged): ``(N_Units, var_Time, D_Features)``
            - Shape (Dense):  ``(Time, D_Features)``
        seq_len (int): Length of the encoder input window (History).
        label_len (int): Length of the "label" or overlap portion prepended to the target.
            Often used as the "Decoder Start Token" in Transformer architectures.
        pred_len (int): Length of the prediction horizon (Future).
        stride (int, optional): Step size between consecutive windows. Defaults to 1.
        pred_offset (int, optional): Temporal gap between the end of the input sequence
            and the start of the prediction. Defaults to 0 (continuous).
        padding_left_flag (bool, optional):
            - ``False`` (Strict): A window is valid only if the unit is long enough
              to provide a full ``seq_len`` history without padding.
            - ``True`` (Padding): Allows windows to start before index 0. Negative
              indices are filled with zeros (Zero-Padding).
        warmup_steps (int, optional): The index in the time series where the **first prediction target**
            should align.
            - If ``None`` (Default): Defaults to ``seq_len``. This implies the first prediction
              occurs immediately after the first full sequence of real data is available.
            - If ``< seq_len`` (and ``padding_left_flag=True``): Inputs will be padded to allow
              predictions to start earlier.

    Examples:
    ---------
    **Scenario 1: Strict Mode (No Padding)**
    Data: ``[A, B, C, D, E]`` (Len 5)
    Config: ``seq_len=3``, ``pred_len=1``, ``padding_left_flag=False``

    - Window 1: Input ``[A, B, C]`` -> Target ``[D]``
    - Window 2: Input ``[B, C, D]`` -> Target ``[E]``
    - Result: 2 valid windows.

    **Scenario 2: Padding Mode (Warmup)**
    Data: ``[A, B, C, D, E]``
    Config: ``seq_len=3``, ``pred_len=1``, ``padding_left_flag=True``, ``warmup_steps=1``



    - **Goal:** First target should be at index 1 ('B').
    - **Logic:** Input ends at index 1. Input starts at ``1 - 3 = -2``.
    - Window 1: Input ``[0, 0, A]`` -> Target ``[B]`` (Padded)
    - Window 2: Input ``[0, A, B]`` -> Target ``[C]`` (Padded)
    - Window 3: Input ``[A, B, C]`` -> Target ``[D]`` (Full Data)
    - Result: Valid windows start earlier, preserving data utility for long-context models.
    """

    def __init__(
        self,
        features: ak.Array,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int = 1,
        pred_offset: int = 0,
        padding_left_flag: int = 0,
        warmup_steps: int = None,
    ):
        self.features = features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.pred_offset = pred_offset
        self.padding_left_flag = padding_left_flag

        # Initialize Warmup Steps
        if warmup_steps is None:
            # Default Strict behavior: First target is immediately after the first full sequence.
            self.warmup_steps = seq_len
        else:
            self.warmup_steps = warmup_steps

        self.device = torch.device("cpu")
        self._checked_output_shapes = False

        assert features.ndim < 4, "Features dim greater than 3 has not been tested"

        # Determine which axis is ragged
        self.ragged_dim = find_singular_ragged_dim({"features": self.features})
        self.is_ragged = True

        if self.ragged_dim is None:
            self.is_ragged = False

        # Get feature dimension
        self.num_features = get_ak_shape(features)[-1]

        # Precompute valid window bounds
        self._indices = np.array(self._build_indices())

    def _build_indices(self) -> List[tuple]:
        """
        Computation of valid (...prefix..., t) tuples.
        """
        required_len = self.seq_len + self.pred_len + self.pred_offset
        stride = int(self.stride)

        # --- Logic for Padding ---
        if self.padding_left_flag:
            # Allow negative starts.
            # Logic: start + seq_len + pred_offset = warmup_steps
            min_start = self.warmup_steps - self.seq_len - self.pred_offset
        else:
            # Strict mode: Input window must be fully inside data boundaries (>= 0).
            min_start = 0

        # --- Path 1: Logic for DENSE arrays (is_ragged=False) ---
        if not self.is_ragged:
            # For a dense array, we treat it as a single time series.
            total_len = len(self.features)

            # Check if data is sufficient. End index of window must be <= total_len
            if (min_start + required_len) > total_len:
                return []

            # Directly calculate the valid start times.
            max_start = total_len - required_len + 1
            starts = range(min_start, max_start, self.stride)

            return [(t,) for t in starts]

        # --- Path 2: Logic for JAGGED arrays ---
        # Get all parent prefixes (no t) for the ragged dimension
        prefixes = ragged_index_tuples(self.features, self.ragged_dim)
        # Precompute segment lengths
        lengths = ak.to_list(ak.num(self.features, axis=self.ragged_dim))

        # Helper to read length given a prefix tuple
        def get_length(prefix):
            if self.ragged_dim == 0:
                # That is potentially dead piece of code.
                return lengths[prefix[0]] if prefix else lengths
            out = lengths
            for p in prefix:
                out = out[p]
            return out

        # Build valid tuples without generating every candidate
        valid: List[tuple] = []
        for prefix in prefixes:
            seg_len = get_length(prefix)

            # Check valid length against min_start
            if (min_start + required_len) <= seg_len:
                max_start = seg_len - required_len + 1
                # Generate only valid starts respecting stride
                starts = range(min_start, max_start, stride)
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

        # Efficient way to create a tuple of 1d arrays for selection
        idx_sel_t = tuple(idx_sel.T)

        # In this case, we have no (ragged) trunk data, only one long time series
        # so we do need to select in the first dimension
        select_in_d0 = len(idx_sel_t) == 1

        # Now point is defined as the coordinate where the sequence starts
        # in the last dimension of the tuple
        now_point = idx_sel_t[-1]

        # Generate relative indices
        x_range = np.arange(0, self.seq_len)[None, :].repeat(b_dim, axis=0)
        y_range = np.arange(
            self.pred_offset, self.label_len + self.pred_len + self.pred_offset
        )[None, :].repeat(b_dim, axis=0)

        # Add 'now_point' to shift to absolute indices
        # x_range may contain negative values if padding is enabled
        x_range = ak.Array(x_range + now_point[:, None])
        y_range = ak.Array(y_range + now_point[:, None] + self.seq_len - self.label_len)

        # --- Data Fetching with Padding Support ---

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

        # 1. Clamp negative indices to 0 for safe fetching. TODO: doublecheck it works as intended
        if self.padding_left_flag:
            # Use numpy max since x_range is structured to support it at this stage
            safe_x_range = np.maximum(x_range, 0)
        else:
            safe_x_range = x_range

        # 2. Fetch Data
        # If indices were negative, we fetched index 0 data temporarily
        raw_x_numpy = leaf_data[safe_x_range].to_numpy()
        raw_y_numpy = leaf_data[y_range].to_numpy()

        seq_x = torch.from_numpy(raw_x_numpy)
        seq_y = torch.from_numpy(raw_y_numpy)

        # 3. Apply Zero Padding (Masking). TODO: doublecheck it works as intended
        if self.padding_left_flag:
            # Create boolean mask where original indices were valid (>= 0)
            valid_mask_x = x_range >= 0

            # Convert mask to tensor and broadcast: (Batch, Seq) -> (Batch, Seq, 1)
            mask_tensor = (
                torch.from_numpy(valid_mask_x.to_numpy()).unsqueeze(-1).to(seq_x.device)
            )

            # Zero out the positions that were originally negative (padded)
            seq_x = seq_x * mask_tensor

        # --- Shape Verification ---
        if not self._checked_output_shapes:
            assert (
                len(seq_x.shape) == 3
            ), f"seq_x ndim mismatch: expected 3, got seq_x.shape {seq_x.shape}"
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
        return self._get_window_for_index(batch_idx)
