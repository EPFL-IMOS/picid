"""Datasource-specific exception hierarchy.

These exceptions let callers distinguish configuration, load, split, and
contract failures without relying on generic assertions.
"""


class DatasourceError(Exception):
    """Base exception for datasource-layer failures."""


class DatasourceConfigurationError(DatasourceError):
    """Raised when datasource configuration is invalid."""


class DatasourceStateError(DatasourceError):
    """Raised when datasource methods are called in the wrong lifecycle state."""


class DatasourceLoadError(DatasourceError):
    """Raised when a datasource fails during load."""


class DatasourceSplitError(DatasourceError):
    """Raised when a datasource fails during split."""


class DatasourceContractError(DatasourceError):
    """Raised when a datasource violates the expected contract."""


class DatasourceCompositionError(DatasourceError):
    """Raised when datasource composition rules are violated."""
