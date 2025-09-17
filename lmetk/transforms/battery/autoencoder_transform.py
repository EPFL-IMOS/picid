import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import hydra
import omegaconf
import torch
from lmetk.transforms.base.base_transform import DenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin
from lmetk.transforms.battery.utils import get_mode_change_right_indices
import pandas as pd
import hydra

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


import torch
from lmetk.data.optimization.sequencer import AutoregressiveSequencer
from lmetk.data.data_objects.data import NamedTransformInput

class AutoencoderTransform(DenseTransform, NoFitMultiSourceTransformMixin):
    """
    Compresses battery cycle data into a mean/std representation using a pre-trained Autoencoder.
    """

    def __init__(self, folder_name: str, key: str="features"):
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
        checkpoint_dir = experiment_dir / "checkpoints"
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
        
        seqs = collect_sequences(0) # Collect seq_x
        
        seqs

        # # Based on the rul column extract the stat
        # target_col = self.target_name

        # cycles_column = data_segment[self.cycles_col].values
        # boundaries = get_mode_change_right_indices(cycles_column)

        # # Get the feature columns
        # feature_columns = data_segment_obj["metadata"]["features_columns"].copy()

        # # Prevent mode columns to be in the features_columns
        # mode_is_present = False
        # if self.mode_col is not None and self.mode_col in feature_columns:
        #     feature_columns.remove(self.mode_col)
        #     mode_is_present = True

        # # Prevent cycles columns to be in the features_columns
        # cycles_is_present = False

        # if self.cycles_col is not None and self.cycles_col in data_segment.columns:
        #     if self.cycles_col in feature_columns:
        #         feature_columns.remove(self.cycles_col)
        #     cycles_is_present = True

        # # Ensure there is no leakage of target data
        # if target_col in feature_columns:
        #     feature_columns.remove(target_col)

        # stats_list = []
        # for i in range(len(boundaries)):
        #     start = 0 if i == 0 else boundaries[i - 1]
        #     end = boundaries[i]
        #     segment = data_segment.iloc[start:end]

        #     assert (
        #         len(segment[target_col].unique()) == 1
        #     ), "Rul should be the same within a segment"
        #     stats = {target_col: segment[target_col].iloc[0]}

        #     if mode_is_present:
        #         assert (
        #             len(segment["mode"].unique()) == 1
        #         ), "Mode should be the same within a segment"
        #         stats["mode"] = segment[self.mode_col].iloc[0]
        #     if cycles_is_present:
        #         assert (
        #             len(segment[self.cycles_col].unique()) == 1
        #         ), "Cycles should be the same within a segment"
        #         stats[self.cycles_col] = segment[self.cycles_col].iloc[0]

        #     for feature in feature_columns:
        #         stats[f"{feature}_mean"] = segment[feature].mean()
        #         stats[f"{feature}_var"] = segment[feature].var()

        #     stats_list.append(stats)

        # data_segment = pd.DataFrame(stats_list)

        return data_segment
