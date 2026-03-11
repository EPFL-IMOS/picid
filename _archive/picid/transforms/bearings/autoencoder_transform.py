import logging
from typing import Any, Dict

import torch
import numpy as np

from picid.transforms.base_transforms.autoencoder_transform import AutoencoderTransform
from picid.data.datasets.rul_context_dataset import RULContextBatchDataset


# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class BearingAutoencoderTransform(AutoencoderTransform):
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

        feat_pre_transform = data[self.key]
        target_pre_transform = data["target"]

        # # Get rid from not needed parts of the moedl
        # del self._model.backbone.decoder_conv
        # del self._model.backbone.decoder_fc

        assert (
            feat_pre_transform.ndim == 2
        ), "Features must be 2D (observartion, features)"

        assert target_pre_transform.ndim == 2, "Target must be 2D (units, features)"

        sq = RULContextBatchDataset(
            data_dict={"features": feat_pre_transform, "rul": target_pre_transform},
            task_type="rul",
            seq_len=self._cfg.dataset.dataset_cfg.seq_len,
            label_len=self._cfg.dataset.dataset_cfg.label_len,
            pred_len=0,
            stride=self._cfg.dataset.dataset_cfg.stride,
        )

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
            batch_x = batch_x.to(self.device)
            with torch.no_grad():
                encoded_batch = self._model.backbone.encode(batch_x).cpu().numpy()
            compressed_seq_x.append(encoded_batch)
            corresponding_rul.append(batch_y)

        features = np.concatenate(compressed_seq_x, axis=0)
        rul = np.concatenate(corresponding_rul, axis=0)

        return {
            "features": features,
            "target": rul,
        }
