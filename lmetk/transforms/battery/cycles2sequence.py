from typing import Any, Dict
import numpy as np
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.datasources.nb14.utils import unflatten_cycles
from lmetk.transforms.base.base_transform import (
    RaggedTransform,
)
from lmetk.transforms.base.multisource import ConcatinatedFitMultiSourceTransformMixin
from lmetk.data.data_objects.data import NamedTransformInput
import logging

logger = logging.getLogger(__name__)


class Cycles2Sequence(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Compress cycles to the mean/std representations of the cycles for NB1 dataset
    """

    def __init__(self, win_len=None, pad_value=0, key="features"):
        super().__init__()
        self.win_len = win_len
        self.key = key

        if pad_value == "np.nan":
            pad_value = np.nan

        self.pad_value = pad_value

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """Transforms a single data segment using the internal ."""

        # Get target if available, otherwise it's None
        flattened_y = data.get("target")

        features, target = unflatten_cycles(
            flattened_x=data[self.key],
            flattened_y=flattened_y,
            cycle_indices=data["metadata"]["cycle_indices"],
            win_len=self.win_len,
            pad_value=self.pad_value,
        )

        logger.info(f"Features shape after convertion to sequences: {features.shape}")

        output = {self.key: features}

        if target is not None:
            # Ensure target is 2D: (N, 1)
            if target.ndim == 3 and target.shape[2] == 1:
                target = target.squeeze(axis=2)
            elif target.ndim == 1:
                target = target.reshape(-1, 1)
            elif target.ndim == 0:
                target = np.array([target]).reshape(-1, 1)
            logger.info(f"Target shape after convertion to sequences: {target.shape}")
            output["target"] = target

        return output

    def fit_data(
        self,
        data: Any,
        metadata: Dict,
    ):

        pass


from lmetk.data.optimization.sequencer import AutoregressiveSequencer

import torch


class Cycles2SequenceRagged(DenseTransform, NoFitMultiSourceTransformMixin):
    def __init__(self, win_len: int, key="features", pad_value=0):

        super().__init__()
        self.seq_len = win_len
        self.label_len = 0
        self.pred_len = win_len
        self.stride = win_len
        self.key = key

        if pad_value == "np.nan":
            pad_value = np.nan

        self.pad_value = pad_value

    def fit_data(self, data: NamedTransformInput, metadata: Dict):
        """This transform do not require fitting."""
        return self

    def transform_data(self, data: NamedTransformInput, metadata: Dict) -> Any:

        features = data[self.key]
        seq = AutoregressiveSequencer(
            features=torch.from_numpy(features),
            time_features=None,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            stride=self.stride,
        )

        def collect_sequences(collect_idx):
            seqs = []
            for i in range(len(seq)):
                results = seq.sequences_batch(i)
                seqs.append(results[collect_idx])
            return torch.stack(seqs, dim=0)

        return data
