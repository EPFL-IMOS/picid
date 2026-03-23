"""Data loading, datasources, datasets, and datamodules."""

from importlib import import_module

__all__ = ["data_objects", "datamodules", "datasources", "datasets"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"picid.data.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
