from typing import Any, Mapping, Sequence
from omegaconf import DictConfig, ListConfig, OmegaConf


def as_seq(x: Any) -> list:
    if isinstance(x, ListConfig):
        x = OmegaConf.to_object(x)  # resolves to plain list
    if not isinstance(x, Sequence) or isinstance(x, (str, bytes)):
        raise TypeError(f"Expected a sequence, got {type(x)}")
    return list(x)


def as_mapping(x: Any) -> dict:
    if isinstance(x, DictConfig):
        x = OmegaConf.to_object(x)
    if not isinstance(x, Mapping):
        raise TypeError(f"Expected a mapping, got {type(x)}")
    return dict(x)
