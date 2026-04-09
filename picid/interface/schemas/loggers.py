from abc import ABC
from typing import Optional

from pydantic import Field, BaseModel



class BaseLogger(ABC, BaseModel):
    """Abstract base class for interface logger configs. Not instantiated directly."""

    name : str
    prefix: str = ''


class WandbLogger(BaseLogger):
    """Logger config for Weights & Biases (wandb).

    Requires a W&B account. The ``entity`` field (team or user name) is
    mandatory. Pass instances in the ``loggers=`` list of
    ``EntryInterface.train()``.

    Parameters
    ----------
    name : str
        Display name for the run in the W&B UI.
    entity : str
        W&B team or user name. Required.
    project : str
        W&B project to log into. Default ``"best_runs"``.
    group : str
        Optional group name for run organisation. Default ``""``.
    tags : list[str]
        Tags attached to the run. Default ``[]``.
    offline : bool
        Log locally without uploading to W&B servers. Default ``False``.
    log_model : bool
        Upload model checkpoints to W&B as artefacts. Default ``False``.

    Examples
    --------
    >>> logger = WandbLogger(name="my_run", entity="my_team", project="phm")
    """

    name : str

    model_class: str = Field('lightning.pytorch.loggers.wandb.WandbLogger',
                                frozen=True, serialization_alias='_target_')

    offline: bool = False
    anonymous: bool = False
    log_model: bool = False

    project: str = 'best_runs'
    group: str = ""
    job_type: str = ""
    tags: list[str] = []

    entity: str


class CsvLogger(BaseLogger):
    """Logger config for PyTorch Lightning's built-in CSV logger.

    Writes metrics to a CSV file in the run output directory. No external
    service required. Pass instances in the ``loggers=`` list of
    ``EntryInterface.train()``.

    Parameters
    ----------
    name : str
        Run name used as the log sub-directory.
    version : str, optional
        Optional version string appended to the log directory name.
        Default ``None``.
    prefix : str
        Prefix prepended to all logged metric keys. Default ``""``.

    Examples
    --------
    >>> logger = CsvLogger(name="my_run")
    >>> logger = CsvLogger(name="my_run", version="v1")
    """

    name: str
    version: Optional[str] = None

    model_class: str = Field('lightning.pytorch.loggers.csv_logs.CSVLogger',
                             frozen=True, serialization_alias='_target_')
