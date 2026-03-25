"""Public datasource base-layer API.

This package exposes the shared contracts, errors, and loader bases that make
up the production datasource architecture. Hydra datasource targets stay
stable in the concrete modules, while this package gives internal code one
clear import surface for the supported base abstractions.
"""

from picid.data.datasources.base.contracts import (
    CompositionDatasourceProtocol,
    DatasourceProtocol,
    LoaderState,
    SplitMode,
)
from picid.data.datasources.base.exceptions import (
    DatasourceCompositionError,
    DatasourceConfigurationError,
    DatasourceStateError,
)
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader
from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.base.predefined_split_loader import (
    PredefinedSplitLoaderBase,
)
from picid.data.datasources.base.single_source_loader import SingleSourceLoader

__all__ = [
    "AbstractDataSourceLoader",
    "CompositionDatasourceProtocol",
    "DatasourceCompositionError",
    "DatasourceConfigurationError",
    "DatasourceProtocol",
    "DatasourceStateError",
    "LoaderState",
    "MultiSourceLoader",
    "PredefinedSplitLoaderBase",
    "SingleSourceLoader",
    "SplitMode",
]
