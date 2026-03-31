import logging

from typing import Any, Dict, OrderedDict

import hydra
from matplotlib.path import Path
import omegaconf
import torch
from torch import nn

from picid.utils.omegaconf_utils import find_config_file

logger = logging.getLogger(__name__)


def correct_state_dict_keys(state_dict: Dict[str, Any]) -> OrderedDict:
    """
    Remove an extraneous ``backbone.`` prefix from checkpoint keys.

    Parameters
    ----------
    state_dict : Dict[str, Any]
        Serialized state dictionary loaded from a checkpoint.

    Returns
    -------
    OrderedDict
        State dictionary with normalized keys.
    """
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        # Use removeprefix for a clean and explicit operation
        new_key = key.removeprefix("backbone.")
        new_state_dict[new_key] = value
    return new_state_dict


def load_model_from_checkpoint(path: Path, device: torch.device) -> nn.Module:
    """
    Load a model from a checkpoint file.

    Parameters
    ----------
    path : Path
        Path to the checkpoint file.
    device : torch.device
        Target device used when loading the checkpoint tensor payload.

    Returns
    -------
    nn.Module
        Instantiated model with checkpoint weights loaded.
    """

    config_path = find_config_file(
        path.parent.parent.parent,
        config_name="config_resolved.yaml",
        select_from_hydra=False,
    )
    cfg = omegaconf.OmegaConf.load(config_path)
    model = hydra.utils.instantiate(cfg.model)

    state_dict = torch.load(path, map_location=device, weights_only=False)["state_dict"]

    # Correct the keys in the state_dict if necessary
    corrected_state_dict = correct_state_dict_keys(state_dict)
    model.load_state_dict(corrected_state_dict)
    logger.info(f"Model loaded successfully from {path}")
    return model
