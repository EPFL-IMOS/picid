import logging
from typing import Any, Dict

import torch
import numpy as np
from tqdm import tqdm

from lmetk.data.optimization.sequencer import RaggedArraySequencer
from lmetk.transforms.base_transforms.autoencoder_transform import AutoencoderTransform

from lmetk.utils.awkward_utils import ak_regularize_regular_axes, blocks_to_ragged_array


# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class AutoencoderTransform(AutoencoderTransform):
    """
    Compresses battery cycle data into a mean/std representation using a pre-trained Autoencoder.
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

        assert (
            feat_pre_transform.ndim == 3
        ), "Features must be 3D (units, cycles, features)"

        assert target_pre_transform.ndim == 2, "Target must be 2D (units, features)"

        sq = RaggedArraySequencer(
            array=feat_pre_transform,
            seq_len=self._cfg.dataset.dataset_cfg.seq_len,
            label_len=self._cfg.dataset.dataset_cfg.label_len,
            pred_len=self._cfg.dataset.dataset_cfg.pred_len,
            stride=self._cfg.dataset.dataset_cfg.stride,
        )

        x, _ = sq.sequences_batch(np.arange(len(sq)))

        res = []
        for start in tqdm(
            range(0, x.shape[0], self.batch_size),
            desc=f"Backbone {self._model.backbone.__class__.__name__} - encoding feature {self.key}",
        ):
            end = min(start + self.batch_size, x.shape[0])
            batch_x = x[start:end].to(self.device)
            with torch.no_grad():
                encoded_batch = self._model.backbone.encode(batch_x).cpu().numpy()
            res.append(encoded_batch)

        res = np.concatenate(res, axis=0)
        index = sq.get_index_array()

        feat = blocks_to_ragged_array(index, blocks=res)
        feat = ak_regularize_regular_axes(feat)

        # Get the target values that correspond to the sequences
        targets = np.sort(np.unique(index[:, 0]))
        targets = target_pre_transform[targets]

        return {
            "features": feat,
            "target": targets,
        }
