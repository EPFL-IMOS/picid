from typing import TYPE_CHECKING

from .datasources import CustomMultiSourceLoader, CustomSingleSourceLoader

if TYPE_CHECKING:
    from .interface import PicidExperiment

__all__ = ["PicidExperiment", "CustomSingleSourceLoader", "CustomMultiSourceLoader"]


def __getattr__(name: str):
    """
    Lazy-load heavy interface dependencies only when needed.

    Parameters
    ----------
    name : str
        Attribute name requested on this package.

    Returns
    -------
    Any
        ``PicidExperiment`` when ``name == "PicidExperiment"``.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy export.
    """
    if name == "PicidExperiment":
        from .interface import PicidExperiment

        return PicidExperiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
