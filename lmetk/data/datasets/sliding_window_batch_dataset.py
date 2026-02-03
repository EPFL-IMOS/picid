import random
import logging

from typing import Dict, List, Optional, Union

import awkward as ak
import numpy as np
import torch

from lmetk.data.datasets.base import BaseDataset
from lmetk.data.datasets.collate_functions import collate_key_value_batch
from lmetk.data.optimization.sequencer import (
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
    Create a separate sequencer for every key in `data_dict`.
    Time features are NOT required.
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
        padding_left_flag: int = 0,
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
                self._seq_idx = self._create_sequence_subset(len(driver), subset_ratio)

            self.sequencers[key] = driver

            # ensure all sequencers have the same length
            if self._length is None:
                self._length = len(driver)
            elif len(driver) != self._length:
                raise ValueError(
                    f"All sequences must yield the same number of windows "
                    f"(mismatch at key '{key}')"
                )

    def _create_sequence_subset(self, sequence_length, subset_ratio):
        """Functionality to test reduced data scenarios."""
        rng = random.Random(self.subset_seed)  # independent RNG
        n_subset = int(sequence_length * subset_ratio)
        seq_idx = rng.sample(range(sequence_length), n_subset)
        seq_idx.sort()
        logger.info(
            f"Subsetting sequences: using {n_subset} out of {sequence_length} sequences."
        )
        logger.info(f"Subset indices start with: {seq_idx[:100]}")
        return np.array(seq_idx)

    def _create_sequencer(self, arr, sequencer_params, key):
        """Create the sequencer object for an array based on the array type."""
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
        The length of this dataset is determined either by the number of sequences,
        or if subset, by the length of that subset.
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
        Returns a dict of {key: (x, y)}.
        For drivers without time features the last two entries are None.
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
