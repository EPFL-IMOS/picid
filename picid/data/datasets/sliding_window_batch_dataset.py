import logging

from typing import Dict, List, Optional, Union

import awkward as ak
import numpy as np
import torch

from picid.data.datasets.base import BaseDataset
from picid.data.datasets.collate_functions import collate_key_value_batch
from picid.data.datasets.subset_sampling import (
    create_sequence_subset,
    make_subset_blocks_indices,
)
from picid.data.optimization.sequencer import (
    BaseSequencer,
    DenseArraySequencer,
    RaggedArraySequencer,
)
import omegaconf

ArrayLike = Union[np.ndarray, torch.Tensor, ak.Array]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlidingWindowBatchDataset(BaseDataset):
    """
    Sliding-window dataset that builds one sequencer per key in ``data_dict``.

    Time features are not required. The dataset supports dense and ragged
    (variable-length) arrays. Each value in ``data_dict`` may be a NumPy
    array, PyTorch tensor, or Awkward array, and a separate sequencer is
    created per key. All sequencers must expose the same number of windows so
    multi-key batches stay aligned. The length formulas below apply to a
    single dense time series.

    Parameters
    ----------
    data_dict : dict[str, ArrayLike]
        Map of array names to arrays. Each value is numpy array, torch tensor,
        or awkward array (dense or ragged). A separate sequencer is created per
        key; all sequencers share the same number of windows (length).
    seq_len : int
        Context window length (number of time steps per input window).
    label_len : int
        Length of the label/decoder context (used by some models).
    pred_len : int
        Prediction horizon (number of time steps to predict).
    stride : int
        Step between consecutive window start indices.
    pred_offset : int, optional
        Offset between context end and prediction start (default 0).
    suffixes : list of str, optional
        Suffixes for context and target keys in the batch (default
        ``["_seq_x", "_seq_y"]``).
    padding_left_flag : bool or int, optional
        If True (default), allow windows to start before index 0 by left-padding
        (min_start = -(seq_len - 1)). If False, only non-padded windows
        (min_start = 0). Affects dataset length (see below).
    warmup_steps : int, optional
        Number of warmup steps for left-padding (overrides default when set).
    subset_ratio : float, optional
        If in (0, 1), use a random subset of window indices for this dataset
        (e.g. for faster training). ``subset_seed`` and ``subset_blocks``
        control sampling.
    subset_seed : int, optional
        Random seed for subset sampling (default 42).
    subset_blocks : int, optional
        When subsetting, cap the number of blocks used (see subset_sampling).
    **kwargs
        Passed through to the base class.

    Notes
    -----
    Dataset length for a dense single time series of length ``T``:

    Let required_len = seq_len + pred_len + pred_offset. Valid window start
    indices form range(min_start, max_start, stride) with
    max_start = T - required_len + 1 (exclusive end). Then
    len(ds) = max(0, (max_start - min_start + stride - 1) // stride).

    - padding_left_flag=False (strict): min_start = 0.
      len(ds) = (T - required_len + stride) // stride.
      Example: T=20, seq_len=2, pred_len=5, stride=1 → required_len=7,
      max_start=14 → len(ds) = 14.

    - padding_left_flag=True (default): min_start = -(seq_len - 1).
      len(ds) = (T - required_len + seq_len + stride - 1) // stride.
      Example: T=20, seq_len=2, pred_len=5, stride=1 → required_len=7,
      range(-1, 14, 1) → len(ds) = 15.
    """

    def __init__(
        self,
        data_dict: Dict[str, ArrayLike],
        seq_len: int,
        label_len: int,
        pred_len: int,
        stride: int,
        pred_offset: int = 0,
        suffixes=["_seq_x", "_seq_y"],
        padding_left_flag: Union[bool, int] = True,
        warmup_steps: int = None,
        subset_ratio: Optional[float] = None,
        subset_seed: int = 42,
        subset_blocks: Optional[int] = None,
        **kwargs,
    ):
        self.sequencers: Dict[str, BaseSequencer] = {}
        self._length: Optional[int] = None
        self._seq_idx = None
        self.suffixes = suffixes
        self.subset_ratio = subset_ratio
        self.subset_seed = subset_seed
        self.subset_blocks = subset_blocks

        # Common params for all sequencers
        sequencer_params = dict(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            stride=stride,
            pred_offset=pred_offset,
            padding_left_flag=padding_left_flag,
            warmup_steps=warmup_steps,
        )

        for key, arr in data_dict.items():
            if arr is None:
                continue

            driver = self._create_sequencer(arr, sequencer_params, key)

            if (
                subset_ratio is not None
                and 0 < subset_ratio < 1
                and self._seq_idx is None
            ):
                len_ds = len(driver)
                if self.subset_blocks is not None and (
                    int(len_ds * subset_ratio) > self.subset_blocks
                ):
                    res = make_subset_blocks_indices(
                        len_ds=len_ds,
                        subset_ratio=subset_ratio,
                        subset_seed=self.subset_seed,
                        subset_blocks=self.subset_blocks,
                    )
                    self._seq_idx = res["seq_idx"]
                else:
                    self._seq_idx = create_sequence_subset(
                        len_ds,
                        subset_ratio=subset_ratio,
                        subset_seed=self.subset_seed,
                    )

                logger.info(
                    f"Subsetting sequences: using {len(self._seq_idx)} out of {len(driver)} sequences."
                )
                logger.info(f"Subset indices start with: {self._seq_idx[:100]}")

            self.sequencers[key] = driver

            # ensure all sequencers have the same length
            if self._length is None:
                self._length = len(driver)
            elif len(driver) != self._length:
                raise ValueError(
                    f"All sequences must yield the same number of windows "
                    f"(mismatch at key '{key}')"
                )

    def _create_sequencer(self, arr, sequencer_params, key):
        """
        Create the sequencer object for an array based on the array type.

        Parameters
        ----------
        arr : np.ndarray | torch.Tensor | ak.Array
            Input array to wrap in a sequencer.
        sequencer_params : dict[str, Any]
            Shared sequencer keyword arguments.
        key : str
            Feature key used for error reporting.

        Returns
        -------
        BaseSequencer
            Dense or ragged sequencer matching the input array type.
        """
        if isinstance(arr, (np.ndarray, torch.Tensor)):
            driver = DenseArraySequencer(arr, **sequencer_params)

        elif isinstance(arr, ak.Array):
            driver = RaggedArraySequencer(arr, **sequencer_params)
        else:
            if isinstance(arr, omegaconf.ListConfig):
                raise NotImplementedError(
                    f"Unsupported type for key '{key}': {type(arr)}. "
                    "Most likely you should use HydraConcat to instantiate a separate iterator for each unit, as it looks like you are trying to past multiunit data into the single unit dataset class."
                )
            else:
                raise NotImplementedError(
                    f"Unsupported type for key '{key}': {type(arr)}"
                )

        return driver

    def __len__(self):
        """
        Return the number of available sequences.

        Returns
        -------
        int
            Number of available sequences or sampled subset size.
        """
        if self._seq_idx is None:
            return self._length or 0
        else:
            return len(self._seq_idx)

    # # Potential improvement: Ensure idx is always a list so the sequencer treats it as a batch
    # def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, tuple]:
    #     # FIX: Ensure idx is always a list for the sequencer
    #     batch_idx = [idx] if isinstance(idx, int) else idx

    #     out = {}
    #     for k, drv in self.sequencers.items():
    #         # Call with list
    #         sequencer_output = drv.sequences_batch(batch_idx)
    #         x, y = sequencer_output

    #         # FIX: Squeeze if input was scalar int
    #         if isinstance(idx, int):
    #             x = x.squeeze(0)
    #             y = y.squeeze(0)

    #         out[f"{k}{self.suffixes[0]}"] = x
    #         out[f"{k}{self.suffixes[1]}"] = y

    #     return out

    def __getitem__(self, idx: Union[int, List[int]]) -> Dict[str, tuple]:
        """
        Return the sequenced tensors for the requested indices.

        Parameters
        ----------
        idx : int | list[int]
            Window index or list of window indices.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing ``<key>_seq_x`` and ``<key>_seq_y`` entries
            for every sequenced input key.
        """
        # Potential improvement: Ensure idx is always a list so the sequencer treats it as a batch
        # If idx is int, wrap it. If list, keep it.
        # batch_idx = [idx] if isinstance(idx, int) else idx

        if self._seq_idx is not None:
            idx = self._seq_idx[idx]

        out = {}
        for k, drv in self.sequencers.items():
            sequencer_output = drv.sequences_batch(idx)
            x, y = sequencer_output

            # seq_x: (B, seq_len, C),
            # seq_y: (B, label_len + pred_len, C);
            # note that y can be empty tensor; (B,T,C)
            out[f"{k}{self.suffixes[0]}"] = x
            out[f"{k}{self.suffixes[1]}"] = y

        return out

    def get_collate_fn(self):
        return collate_key_value_batch
