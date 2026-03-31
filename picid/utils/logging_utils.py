"""Utilities for logging hyperparameters."""

from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from picid.utils import pylogger


log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def get_hydra_override(key: str):
    if not HydraConfig.initialized():
        return None
    for override in HydraConfig.get().overrides.task:
        if override.startswith(f"{key}="):
            return override.split("=", 1)[1]
    return None


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    r"""
    Control which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    Parameters
    ----------
    object_dict : dict[str, Any]
        A dictionary containing the following objects:
            - `"cfg"`: A DictConfig object containing the main config.
            - `"model"`: The Lightning model.
            - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    exp_name = get_hydra_override("experiment")
    cfg["exp_name"] = exp_name
    # lgr.experiment.config.update(cfg)
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # save number of model parameters
    hparams["model/params/total"] = object_dict["model/params/total"]
    hparams["model/params/trainable"] = object_dict["model/params/trainable"]

    for key in cfg:
        hparams[key] = cfg[key]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
