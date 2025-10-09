import logging
from pathlib import Path
from typing import Any, Dict


import hydra
import omegaconf
import torch

from lmetk.transforms.base.base_transform import RaggedOrDenseTransform
from lmetk.transforms.base.multisource import NoFitMultiSourceTransformMixin

from lmetk.utils.omegaconf_utils import find_config_file
from lmetk.utils.torch_utils import correct_state_dict_keys

# Set up a logger for better feedback
logger = logging.getLogger(__name__)


class AutoencoderTransform(
    RaggedOrDenseTransform,
    NoFitMultiSourceTransformMixin,
):
    """
    Compresses battery  data into a mean/std representation using a pre-trained Autoencoder.
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
        raise NotImplementedError("This transform is not implemented yet.")
