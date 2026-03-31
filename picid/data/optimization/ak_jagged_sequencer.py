import logging
from typing import List, Optional, Tuple, Union

import awkward as ak
import numpy as np
import torch

from picid.utils.awkward_utils import (
    find_singular_ragged_dim,
    get_ak_shape,
    ragged_index_tuples,
)

logger = logging.getLogger(__name__)


class AkwardJaggedAutoregressiveSequencer:
    """
    Autoregressive sequencer for *ragged* (variable-length) Awkward Arrays.

    This class generates overlapping input/output sliding windows from
    time-series data. It is designed to handle multi-unit jagged data, where
    each unit (for example an engine or a battery cycle) has a different
    temporal length, and can also handle dense single-series arrays.

    Parameters
    ----------
    features : ak.Array
        Input data array.
        - Shape (Ragged): ``(N_Units, var_Time, D_Features)``
        - Shape (Dense):  ``(Time, D_Features)``
    seq_len : int
        Length of the encoder input window (History).
    label_len : int
        Length of the "label" or overlap portion prepended to the target.
        Often used as the "Decoder Start Token" in Transformer architectures.
    pred_len : int
        Length of the prediction horizon (Future).
    stride : int, optional
        Step size between consecutive windows. Defaults to 1.
    pred_offset : int, optional
        Temporal gap between the end of the input sequence
        and the start of the prediction. Defaults to 0 (continuous).
    padding_left_flag : Union[bool, int], optional
        - ``False`` or ``0`` (Strict): A window is valid only if the unit is long enough
          to provide a full ``seq_len`` history without padding.
        - ``True`` or ``1`` (Padding - Default): Allows windows to start before index 0. Negative
          indices are filled according to ``padding_mode``.
    padding_mode : str, optional
        Defines the padding strategy when ``padding_left_flag=True``.
        - ``"edge"`` (Default): Fills padded indices with the first available observation (Index 0).
          This effectively repeats the first time step.
        - ``"zero"``: Fills padded indices with zeros.
    warmup_steps : int, optional
        Controls the start index of the first window.
        - **If padding_left_flag=True**: Represents the **number of padded steps** to include in the first window.
          - Default: ``seq_len - 1``. This maximizes padding such that the first window contains
            only **1 real observation** at the end.
          - Value ``0``: Starts at index 0 (no padding).
        - **If padding_left_flag=False**: Represents the **index offset** for the start of the first sequence.
          - Default: ``0``. This means the window starts at index 0 (capturing the first full valid sequence `0:seq_len`).

    Notes
    -----
    Key features:

    1. Iterates over the first dimension for jagged multi-unit data and slices
       windows relative to the start of each unit.
    2. Supports strict windowing where input sequences consist entirely of
       real data and short units are dropped.
    3. Supports left padding so predictions can start earlier in the time
       series with zeros or edge values.

    For a given anchor time step ``t``:

    - Input ``seq_x`` is ``[t - seq_len : t]``.
    - Target ``seq_y`` is
      ``[t + pred_offset : t + pred_offset + label_len + pred_len]``.

    Examples
    --------
    **Scenario 1: Default Padding (Edge)**
    Data: ``[A, B, C, D]``, ``seq_len=3``
    Default ``warmup_steps = seq_len - 1 = 2``.

    - Window 1 (Start -2): ``[A, A, A]`` (2 pads, 1 real) -> Target ``[B]``
    - Window 2 (Start -1): ``[A, A, B]`` (1 pad, 2 real) -> Target ``[C]``
    - Window 3 (Start 0):  ``[A, B, C]`` (0 pads, 3 real) -> Target ``[D]``

    **Scenario 2: Manual Warmup**
    Config: ``warmup_steps=0`` (padding_left_flag=True)

    - Window 1 (Start 0): ``[A, B, C]`` -> Target ``[D]`` (Padding is skipped)
    """

    def __init__(
        self,
        features: ak.Array,
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int = 1,
        pred_offset: int = 0,
        padding_left_flag: Union[bool, int] = True,
        padding_mode: str = "edge",
        warmup_steps: int = None,
    ):
        self.features = features
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.pred_offset = pred_offset

        # Validate and Normalize padding_left_flag
        # Check strict 0/1 constraint if it is an integer (bools are ints in Python,
        # so True/False pass "in [0, 1]" check automatically)
        if isinstance(padding_left_flag, int) and padding_left_flag not in [0, 1]:
            raise ValueError(
                f"padding_left_flag must be boolean or int (0 or 1), got {padding_left_flag}"
            )

        # Normalize to boolean
        self.padding_left_flag = bool(padding_left_flag)

        if padding_mode not in ["zero", "edge"]:
            raise ValueError(
                f"padding_mode must be 'zero' or 'edge', got '{padding_mode}'"
            )
        self.padding_mode = padding_mode

        # --- Logic for Padding ---
        # Initialize Warmup Steps (Indices Logic)
        if self.padding_left_flag:
            if warmup_steps is None:
                # index_start_offset represents the MAGNITUDE of the index start.
                # E.g. warmup_steps=-2 -> min_start = -2. in _build_indices()

                # Default: Start with maximum padding (just 1 real data point)
                self.index_start_offset = -(seq_len - 1)
            else:
                self.index_start_offset = -warmup_steps
                logger.info(
                    f"Padding enabled with explicit warmup_steps={-self.index_start_offset}. "
                    f"This defines the number of padded steps. "
                    f"The first input window will start at index -{self.index_start_offset}."
                )
        else:
            # Strict mode: Input window must be fully inside data boundaries (>= 0).
            # warmup_steps here acts as an additional positive offset if set.
            # Strict mode: Default to 0 (Start at first full sequence)
            if warmup_steps is None:
                self.index_start_offset = 0
            else:
                self.index_start_offset = warmup_steps

        self.device = torch.device("cpu")
        self._checked_output_shapes = False

        assert features.ndim < 4, "Features dim greater than 3 has not been tested"

        # Determine which axis is ragged
        self.ragged_dim = find_singular_ragged_dim(self.features)
        self.is_ragged = True

        if self.ragged_dim is None:
            self.is_ragged = False

        # Get feature dimension
        self.num_features = get_ak_shape(features)[-1]

        # Precompute valid window bounds
        self._indices = np.array(self._build_indices())

    def _build_indices(self) -> List[tuple]:
        """
        Compute valid prefix/time index tuples for all windows.

        Returns
        -------
        list[tuple]
            Valid index tuples for all windows.
        """
        required_len = self.seq_len + self.pred_len + self.pred_offset
        stride = int(self.stride)

        min_start = self.index_start_offset

        # --- Path 1: Logic for DENSE arrays (is_ragged=False) ---
        if not self.is_ragged:
            # For a dense array, we treat it as a single time series.
            total_len = len(self.features)

            # Check if data is sufficient. End index of window must be <= total_len
            if (min_start + required_len) > total_len:
                logger.warning(
                    f"Attempting to create a sequencer with seq_len: {self.seq_len} pred_len: {self.pred_len} and pred_offset: {self.pred_offset}. "
                    f"This requires the length of the sequenced dimension to be at least {required_len} "
                    f"but the features you provided are of length: {total_len},"
                    "resulting in an empty sequencer."
                )
                return []
            else:
                # Directly calculate the valid start times.
                max_start = total_len - required_len + 1
                starts = range(min_start, max_start, self.stride)

                return [(t,) for t in starts]

        else:
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
            n_prefixes = len(prefixes)
            for id_pfx, prefix in enumerate(prefixes):
                seg_len = get_length(prefix)

                # Check valid length against min_start
                if (min_start + required_len) <= seg_len:
                    max_start = seg_len - required_len + 1
                    # Generate only valid starts respecting stride
                    starts = range(min_start, max_start, stride)
                    valid.extend([(*prefix, t) for t in starts])
                else:
                    logger.warning(
                        f"Attempting to create a ragged sequencer for dense leaf data {id_pfx}/{n_prefixes} "
                        f"(seq_len: {self.seq_len} pred_len: {self.pred_len} and pred_offset: {self.pred_offset}). "
                        f"This requires the length of the sequenced dimension to be at least {required_len} "
                        f"but the segment length is {seg_len}, resulting in an empty sequencer."
                    )

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

        # 1. Clamp negative indices to 0 for safe fetching.
        #    If padding_mode='edge', this step ALONE achieves the result:
        #    negative indices fetch index 0 (repeating the first value).
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

        # 3. Apply Zero Padding (Masking)
        #    Only run this if mode is "zero".
        #    If mode is "edge", we keep the values clamped in step 1.
        if self.padding_left_flag and self.padding_mode == "zero":
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
        """Return a batch of input and target windows for sequencer indices.

        Parameters
        ----------
        batch_idx
            Integer indices into the precomputed window index table returned
            by ``get_index_array()``.

        Returns
        -------
        A tuple ``(seq_x, seq_y)`` where:

        - ``seq_x`` has shape ``(batch, seq_len, num_features)`` and contains the
          encoder/history window.
        - ``seq_y`` has shape ``(batch, label_len + pred_len, num_features)`` and
          contains the decoder overlap plus forecast horizon.

        Notes
        -----
        When left padding is enabled, early windows may include padded history in
        ``seq_x``. Padding uses the configured ``padding_mode``; ``seq_y`` always
        contains real observations from the source segment.
        """
        return self._get_window_for_index(batch_idx)
