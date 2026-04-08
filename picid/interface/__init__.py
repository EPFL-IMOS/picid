from .interface import EntryInterface
from .datasources import CustomSingleSourceLoader, CustomMultiSourceLoader

__all__ = ["EntryInterface", "CustomSingleSourceLoader", "CustomMultiSourceLoader"]


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
        ``EntryInterface`` when ``name == "EntryInterface"``.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy export.
    """
    if name == "EntryInterface":
        from .interface import EntryInterface

        return EntryInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
