import torch
import random
import numpy as np
import json

import logging
from omegaconf import ListConfig

logger = logging.getLogger(__name__)


def set_reproducibility(seed=42, deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class StandardScaler:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = (
            torch.from_numpy(self.mean).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.mean
        )
        std = (
            torch.from_numpy(self.std).type_as(data).to(data.device)
            if torch.is_tensor(data)
            else self.std
        )
        return (data * std) + mean


def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)
    return args


def string_split(str_for_split):
    str_no_space = str_for_split.replace(" ", "")
    str_split = str_no_space.split(",")
    value_list = [eval(x) for x in str_split]

    return value_list


def key_exists_anywhere(data, target_key):
    # Check if object has dict-like methods
    if hasattr(data, "keys") and hasattr(data, "values") and hasattr(data, "items"):
        # First check if the target key exists at this level
        if target_key in data:
            return True, data[target_key]

        # Then recursively search in all values
        for value in data.values():
            found, val = key_exists_anywhere(value, target_key)
            if found:
                return found, val

    elif isinstance(data, (list, tuple)):
        for item in data:
            found, val = key_exists_anywhere(item, target_key)
            if found:
                return found, val

    return False, None
