import os
import json

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


class ModelCheckpointWithConfig(ModelCheckpoint):
    """
    A custom ModelCheckpoint callback that also saves a configuration dictionary
    to a JSON file in the checkpoint directory.

    This callback saves the config file only once when the first checkpoint is created.

    Parameters
    ----------
    config : Dict[str, Any]
        A dictionary containing the configuration/hyperparameters.
    config_filename : str
        The name of the configuration file to be saved.
        Defaults to "hparams.json".
    *args :
        Positional arguments to pass to the parent ModelCheckpoint.
    **kwargs :
        Keyword arguments to pass to the parent ModelCheckpoint.
    """

    def __init__(self, config_filename: str = "hparams.json", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = None
        self.config_filename = config_filename

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # First, let the parent class save the model checkpoint
        super()._save_checkpoint(trainer, filepath)

        # On the main process (global_rank == 0), save the config file.
        # We check if the file already exists to avoid writing it repeatedly.
        if trainer.is_global_zero:
            dirpath = os.path.dirname(filepath)
            config_path = os.path.join(dirpath, self.config_filename)

            assert self.config is not None, "Config is not set."
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    json.dump(self.config, f, indent=4)
                if self.verbose:
                    trainer.print(f"Configuration saved to {config_path}")
