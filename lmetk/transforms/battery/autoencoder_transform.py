import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import awkward as ak
import hydra
import omegaconf
import torch
from tqdm import tqdm

from lmetk.data.optimization.sequencer import RaggedArraySequencer
from lmetk.transforms.base.base_transform import RaggedTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices
from lmetk.utils.awkward_utils import ak_regularize_regular_axes, blocks_to_ragged_array
import numpy as np

from lmetk.utils.omegaconf_utils import find_config_file
from lmetk.utils.torch_utils import correct_state_dict_keys

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class AutoencoderTransform(RaggedTransform, NoFitMultiSourceTransformMixin):
    """
    Compresses battery cycle data into a mean/std representation using a pre-trained Autoencoder.
    """

    def __init__(self, folder_name: str, key: str = "features", batch_size: int = 2048):
        """
        Initializes the transform by loading the model configuration and weights.

        Args:
            folder_name: The path to the experiment folder containing the model checkpoint
                         and configuration file.
        """
        super().__init__()
        self.key = key
        self.batch_size = batch_size
        experiment_dir = Path(folder_name)
        if not experiment_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment directory not found at: {experiment_dir}"
            )

        self._cfg = None
        self._model = self._load_model(experiment_dir)
        # Getr the device:
        if self._cfg.trainer.accelerator == "cpu":
            self.device = "cpu"
        else:
            self.device = self._cfg.trainer.devices[-1]

        self._model.to(self.device)
        self._model.eval()  # Set model to evaluation mode
        self._model.backbone.eval()

        self.seq_len = self._cfg.task_definition.seq_len
        self.label_len = self._cfg.task_definition.label_len
        self.pred_len = self._cfg.task_definition.pred_len
        self.stride = self.pred_len

    def _load_model(self, experiment_dir: Path) -> torch.nn.Module:
        """Loads the model from the specified experiment directory."""
        config_path = find_config_file(experiment_dir)
        self._cfg = omegaconf.OmegaConf.load(config_path)

        model = hydra.utils.instantiate(self._cfg.model)

        checkpoint_path = self._find_checkpoint_file(experiment_dir)
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["state_dict"]

        # Correct the keys in the state_dict if necessary
        corrected_state_dict = correct_state_dict_keys(state_dict)
        model.load_state_dict(corrected_state_dict)

        logger.info(f"Successfully loaded model from {checkpoint_path}")
        return model

    @staticmethod
    def _find_checkpoint_file(experiment_dir: Path) -> Path:
        """Finds the unique .ckpt file in the checkpoints subdirectory."""
        checkpoint_dir = (
            experiment_dir / "checkpoints/checkpoints"
        )  # TODO: FIX WEIRD PATH
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory not found in {experiment_dir}"
            )

        try:
            # Use a generator expression and next() for an efficient single-item search
            checkpoint_path = next(checkpoint_dir.glob("*.ckpt"))
        except StopIteration:
            raise FileNotFoundError(
                f"No .ckpt files found in {checkpoint_dir}"
            ) from None

        # Ensure no other checkpoints exist
        if list(checkpoint_dir.glob("*.ckpt"))[1:]:
            logger.warning(
                f"Multiple checkpoints found in {checkpoint_dir}. Using {checkpoint_path}."
            )

        return checkpoint_path

    def fit_data(self, data: Any, metadata: Dict):
        """This transform does not require fitting."""
        pass

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

        # TODO: Later we will need to better now to workout this
        # concat_dataset_over_cycles = HydraConcatDataset(
        #     dataset_cfg=self._cfg.dataset.dataset_cfg,
        #     data_dict={self.key: [i for i in data[self.key]]},
        #     meta_data_dict=None,
        # )
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

        # datasets_cycles = concat_dataset_over_cycles.datasets

        # res, idx_valid = [], []
        # for idx, dataset in enumerate(datasets_cycles):
        #     dataset_length = dataset.__len__()
        #     if dataset_length > 0:
        #         batch_idx = np.arange(dataset_length)
        #         all_data = dataset.__getitem__(batch_idx)[f"{self.key}_seq_x"]
        #         # Go over the all_data in the batch and extract cycles

        #         dataset_cycles_embeds = []
        #         for start in range(0, all_data.shape[0], self.batch_size):
        #             end = min(start + self.batch_size, all_data.shape[0])
        #             cycle_seqs = all_data[start:end].to(self.device)
        #             # Encode the cycle sequences
        #             with torch.no_grad():
        #                 dataset_cycles_embeds.append(
        #                     self._model.backbone.encode(cycle_seqs).cpu().numpy()
        #                 )

        #         # Collect embeddings for all cycles in this dataset
        #         compressed_data = np.concatenate(dataset_cycles_embeds, axis=0)
        #         res.append(torch.tensor(compressed_data))
        #         assert (
        #             all_data.shape[0] == compressed_data.shape[0]
        #         ), "Length mismatch after compression"

        #         # Collect idx that has successfully passed the transformations
        #         idx_valid.append(idx)

        return {
            "features": feat,
            "target": targets,
        }
