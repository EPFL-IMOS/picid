from abc import ABC
from typing import Optional

from pydantic import Field, BaseModel



class BaseLogger(ABC, BaseModel):
    name : str
    prefix: str = ''


class WandbLogger(BaseLogger):
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
    name: str
    version: Optional[str] = None

    model_class: str = Field('lightning.pytorch.loggers.csv_logs.CSVLogger',
                             frozen=True, serialization_alias='_target_')
