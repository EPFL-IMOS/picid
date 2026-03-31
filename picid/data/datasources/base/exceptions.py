"""Datasource-specific exception hierarchy.

These exceptions let callers distinguish configuration, load, split, and
contract failures without relying on generic assertions.
"""


class DatasourceError(Exception):
    """
    Define the Datasource Error helper.

    """


class DatasourceConfigurationError(DatasourceError):
    """
    Define the Datasource Configuration Error helper.

    """


class DatasourceStateError(DatasourceError):
    """Raised when datasource methods are called in the wrong lifecycle state."""


class DatasourceLoadError(DatasourceError):
    """
    Define the Datasource Load Error helper.

    """


class DatasourceSplitError(DatasourceError):
    """
    Define the Datasource Split Error helper.

    """


class DatasourceContractError(DatasourceError):
    """
    Define the Datasource Contract Error helper.

    """


class DatasourceCompositionError(DatasourceError):
    """
    Define the Datasource Composition Error helper.

    """
