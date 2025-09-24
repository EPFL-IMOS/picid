import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import hydra
import omegaconf
import torch
from lmetk.transforms.base.base_transform import DenseTransform, RaggedTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices
import pandas as pd
import hydra

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


import torch
from lmetk.data.optimization.sequencer import AutoregressiveSequencer
from lmetk.data.data_objects.data import NamedTransformInput


class AutoencoderTransform(RaggedTransform, NoFitMultiSourceTransformMixin):
    """
    Compresses battery cycle data into a mean/std representation using a pre-trained Autoencoder.
    """

    def __init__(self, folder_name: str, key: str = "features"):
        """
        Initializes the transform by loading the model configuration and weights.

        Args:
            folder_name: The path to the experiment folder containing the model checkpoint
                         and configuration file.
        """
        super().__init__()
        self.key = key
        experiment_dir = Path(folder_name)
        if not experiment_dir.is_dir():
            raise FileNotFoundError(
                f"Experiment directory not found at: {experiment_dir}"
            )

        self._cfg = None
        self._model = self._load_model(experiment_dir)
        self._model.eval()  # Set model to evaluation mode

        self.seq_len = self._cfg.task_definition.seq_len
        self.label_len = self._cfg.task_definition.label_len
        self.pred_len = self._cfg.task_definition.pred_len
        self.stride = self.pred_len

    def _load_model(self, experiment_dir: Path) -> torch.nn.Module:
        """Loads the model from the specified experiment directory."""
        config_path = self._find_config_file(experiment_dir)
        self._cfg = omegaconf.OmegaConf.load(config_path)

        model = hydra.utils.instantiate(self._cfg.model)

        checkpoint_path = self._find_checkpoint_file(experiment_dir)
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["state_dict"]

        # Correct the keys in the state_dict if necessary
        corrected_state_dict = self._correct_state_dict_keys(state_dict)
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

    @staticmethod
    def _find_config_file(experiment_dir: Path) -> Path:
        """Finds the config.yaml file, excluding .hydra directories."""
        # This searches for 'config.yaml' in all subdirectories of experiment_dir
        for config_path in experiment_dir.rglob("config.yaml"):
            # Ensure the config is inside a .hydra metadata folder
            if ".hydra" in config_path.parts:
                return config_path

        raise FileNotFoundError(
            f"config.yaml not found in {experiment_dir} (excluding .hydra directories)"
        )

    @staticmethod
    def _correct_state_dict_keys(state_dict: Dict[str, Any]) -> OrderedDict:
        """Removes an extraneous 'backbone.' prefix from state_dict keys."""
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # Use removeprefix for a clean and explicit operation
            new_key = key.removeprefix("backbone.")
            new_state_dict[new_key] = value
        return new_state_dict

    def fit_data(self, data: Any, metadata: Dict):
        """This transform does not require fitting."""
        pass

    def transform_data(self, data: Any, metadata: Dict) -> Any:
        """
        Transforms a single data segment using the loaded autoencoder model.

        Note: The implementation for data transformation needs to be added here.
        """
        features = data[self.key]

        concat_dataset_over_cycles = hydra.utils.instantiate(
            self._cfg.dataset,
            data_dict={self.key: [i for i in data[self.key]]},
            meta_data_dict=None,
        )

        datasets_cycles = concat_dataset_over_cycles.datasets

        def collect_sequences(collect_idx):
            seqs = []
            for i in range(len(dataset)):
                results = dataset[i]
                seqs.append(results[self.key][collect_idx])
            return torch.cat(seqs, dim=0)

        res = []
        for dataset in datasets_cycles:
            assert (
                dataset.__len__() != 0
            ), "Dataset is empty. Meaning we will loose some cycle"
            if dataset.__len__() > 0:
                cycle_seqs = collect_sequences(0)  # Collect seq_x
                res.append(self._model.backbone.encode(cycle_seqs))

        print()

        return
