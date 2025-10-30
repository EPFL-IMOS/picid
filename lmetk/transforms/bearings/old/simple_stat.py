import logging
from typing import Any, Dict

import torch
import numpy as np
from tqdm import tqdm

from lmetk.data.optimization.sequencer import RaggedArraySequencer
from lmetk.transforms.base.base_transform import RaggedOrDenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.data.datasets.rul_context_dataset import RULContextBatchDataset
from lmetk.utils.awkward_utils import ak_regularize_regular_axes, blocks_to_ragged_array


# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class BearingStat(RaggedOrDenseTransform, NoFitMultiSourceTransformMixin):
    """
    Compresses bearing data into representation using a pre-trained Autoencoder.
    """

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """
        Transforms a single data segment using the loaded autoencoder model.

        Note: The implementation for data transformation needs to be added here.
        """
        assert "features" in data, "Features key is not in data"
        assert "target" in data, "Target is not in data"

        feat_pre_transform = data["features"]
        target_pre_transform = data["target"]

        assert (
            feat_pre_transform.ndim == 2
        ), "Features must be 2D (observartion, features)"

        assert target_pre_transform.ndim == 2, "Target must be 2D (units, features)"

        sq = RULContextBatchDataset(
            data_dict={"features": feat_pre_transform, "rul": target_pre_transform},
            task_type="rul",
            seq_len=64,
            label_len=0,
            pred_len=0,
            stride=1,
        )
        self.batch_size = 65536
        # Process sequences in batches to avoid memory issues
        indices = np.arange(len(sq))
        compressed_seq_x = []
        corresponding_rul = []
        batch_indices = range(0, len(indices), self.batch_size)
        for start in batch_indices:
            end = min(start + self.batch_size, len(indices))
            batch_idx = indices[start:end]
            batch = sq.__getitem__(batch_idx)
            batch_x = batch["features"]
            batch_y = batch["rul"]
            batch_x = batch_x
            with torch.no_grad():
                encoded_batch = [
                    np.mean(batch_x.cpu().numpy(), axis=1),
                    np.std(batch_x.cpu().numpy(), axis=1),
                    np.max(batch_x.cpu().numpy(), axis=1),
                    np.min(batch_x.cpu().numpy(), axis=1),
                ]
            compressed_seq_x.append(np.concatenate(encoded_batch, axis=1))
            corresponding_rul.append(batch_y)

        features = np.concatenate(compressed_seq_x, axis=0)
        rul = np.concatenate(corresponding_rul, axis=0)

        return {
            "features": features,
            "target": rul,
        }
