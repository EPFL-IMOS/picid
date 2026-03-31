"""
Expose datasource helpers for the datasources package.

"""

from importlib import import_module

__all__ = [
    "AbstractDataSourceLoader",
    "MultiSourceLoader",
    "PHMDMultiSourceLoader",
    "SingleSourceLoader",
]

_LAZY_IMPORTS = {
    "AbstractDataSourceLoader": (
        "picid.data.datasources.base.interfaces",
        "AbstractDataSourceLoader",
    ),
    "MultiSourceLoader": (
        "picid.data.datasources.base.multi_source_loader",
        "MultiSourceLoader",
    ),
    "PHMDMultiSourceLoader": (
        "picid.data.datasources.base.phmd_loader",
        "PHMDMultiSourceLoader",
    ),
    "SingleSourceLoader": (
        "picid.data.datasources.base.single_source_loader",
        "SingleSourceLoader",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
