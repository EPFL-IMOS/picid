# @override decorator
import torch
from typing import Dict, List, override

from lightning_fabric.utilities.data import AttributeDict

from lmetk.data.datasets.collate_functions import collate_key_value_batch
from lmetk.data.datasets.context_dataset import ContextBatchDataset


class RULContextBatchDataset(ContextBatchDataset):
    """
    Dataset for RUL prediction with context features.
    Combines target RUL sequences with context feature sequences.
    """

    @override
    def __getitem__(self, batch_idx: List[int]):
        target: Dict = self.target_dataset.__getitem__(batch_idx)
        context: Dict = self.context_dataset.__getitem__(batch_idx)
        target_seq_x = target["rul_seq_x"]
        context_seq_x = context["features_seq_x"]

        d = AttributeDict(
            {
                "batch_idx": torch.tensor(batch_idx),
                # (B=1, seq_len, C) -> (B=1, C),
                # take the last time step,
                # collate fn will concatenate over the batch dim
                "rul": target_seq_x[:, -1],
                # (B, seq_len, C)
                "features": context_seq_x,
            }
        )

        return AttributeDict(d)

    def get_collate_fn(self):
        return collate_key_value_batch
