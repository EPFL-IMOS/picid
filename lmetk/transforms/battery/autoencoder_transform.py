import logging
from typing import Any, Dict, Tuple

import torch
import numpy as np
import awkward as ak
from tqdm import tqdm

from lmetk.data.optimization.sequencer import RaggedArraySequencer
from lmetk.transforms.base_transforms.autoencoder_transform import (
    BaseAutoencoderTransform,
)
from lmetk.utils.awkward_utils import ak_regularize_regular_axes, blocks_to_ragged_array

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class AutoencoderTransform(BaseAutoencoderTransform):
    """
    Compresses battery cycle data into a mean/std representation using a
    pre-trained Autoencoder.
    """

    def _validate_input_data(self, data: Dict[str, Any]) -> Tuple[Any, Any]:
        """Validates input data and extracts raw features and targets."""
        assert "features" in data, "Features key is not in data"
        assert "target" in data, "Target is not in data"

        feat_pre_transform = data[self.key]
        target_pre_transform = data["target"]

        assert (
            feat_pre_transform.ndim == 3
        ), "Features must be 3D (units, cycles, features)"

        # Original code commented this out, preserving that choice.
        # assert target_pre_transform.ndim == 2, "Target must be 2D (units, features)"

        return feat_pre_transform, target_pre_transform

    def _create_sequencer(self, array: Any, cfg: Any) -> RaggedArraySequencer:
        """Creates a RaggedArraySequencer with standardized config."""
        kwargs = dict(
            array=array,
            seq_len=cfg.seq_len,
            label_len=cfg.label_len,
            pred_len=cfg.pred_len,
            stride=cfg.stride,
        )
        if hasattr(cfg, "pred_offset"):
            kwargs["pred_offset"] = cfg.pred_offset
        if hasattr(cfg, "padding_left_flag"):
            kwargs["padding_left_flag"] = cfg.padding_left_flag
        return RaggedArraySequencer(**kwargs)

    def _process_target_batch(self, y_batch: np.ndarray) -> np.ndarray:
        """
        Processes a single target batch, ensuring it has the correct
        shape by taking the last time step.
        """
        if y_batch.ndim == 3:
            # ensure that all the values within the 1st dim are the same
            assert np.all(
                y_batch[:, :, 0] == y_batch[:, 0, 0][:, np.newaxis]
            ), "Target values within the sequence are not the same"
            # Take the last time step
            y_batch = y_batch[:, -1, :]
        elif y_batch.ndim == 2:
            # Already in the correct shape (or implies last time step)
            y_batch = y_batch[:, -1]
        else:
            raise ValueError(f"Target batch has incorrect dimensions: {y_batch.ndim}")
        return y_batch

    def _reconstruct_ragged_array(self, index: Any, blocks: np.ndarray) -> Any:
        """Reconstructs and regularizes a ragged array from blocks."""
        ragged_arr = blocks_to_ragged_array(index, blocks=blocks)
        return ak_regularize_regular_axes(ragged_arr)

    def _process_in_batches(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the main batch processing loop for encoding features
        and processing targets.
        """
        res, res_target = [], []

        desc = (
            f"Backbone {self._model.backbone.__class__.__name__} "
            f"- encoding feature {self.key}"
        )

        for start in tqdm(range(0, x.shape[0], self.batch_size), desc=desc):
            end = min(start + self.batch_size, x.shape[0])

            # 1. Encode features
            batch_x = x[start:end].to(self.device)
            with torch.no_grad():
                encoded_batch = self._model.backbone.encode(batch_x).cpu().numpy()
            res.append(encoded_batch)

            # 2. Process targets
            y_batch = y[start:end].numpy()
            processed_target = self._process_target_batch(y_batch)
            res_target.append(processed_target)

        return np.concatenate(res, axis=0), np.concatenate(res_target, axis=0)

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """
        Transforms a single data segment using the loaded autoencoder model.
        """
        # 1. Validate and extract raw data
        features_raw, target_raw = self._validate_input_data(data)

        # 2. Create sequences
        dataset_cfg = self._cfg.dataset.dataset_cfg
        feature_sequencer = self._create_sequencer(features_raw, dataset_cfg)
        target_sequencer = self._create_sequencer(target_raw, dataset_cfg)

        assert len(feature_sequencer) == len(target_sequencer), (
            "Feature and target sequencers produced a different number of "
            f"samples: {len(feature_sequencer)} vs {len(target_sequencer)}"
        )

        # 3. Get all sequence batches (as Tensors)
        x, _ = feature_sequencer.sequences_batch(np.arange(len(feature_sequencer)))
        y, _ = target_sequencer.sequences_batch(np.arange(len(target_sequencer)))

        # 4. Process all batches
        encoded_features, processed_targets = self._process_in_batches(x, y)

        # 5. Validate processed shapes
        assert encoded_features.shape[:-1] == processed_targets.shape[:-1], (
            "Encoded features and target have different shapes: "
            f"{encoded_features.shape} vs {processed_targets.shape}"
        )

        # 6. Reconstruct ragged arrays
        index = feature_sequencer.get_index_array()
        feat = self._reconstruct_ragged_array(index, encoded_features)
        targets = self._reconstruct_ragged_array(index, processed_targets)

        return {
            "features": feat,
            "target": targets,
        }
