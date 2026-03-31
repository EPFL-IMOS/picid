"""Abstract splitter contracts for datasource preprocessing."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    from picid.data.datasources.base.contracts import DatasourceProtocol


class SourceSplitter(ABC):
    """
    Interface for splitters that partition whole datasource sources.

    The splitter contract stays structural so preprocessing can work with any
    datasource implementation that satisfies the shared protocol.
    """

    @abstractmethod
    def split_data(
        self, sources: Dict[str, "DatasourceProtocol"]
    ) -> Dict[str, Dict[str, "DatasourceProtocol"]]:
        """
        Split datasource loaders into train/val/test source groups.

        Parameters
        ----------
        sources : Dict[str, "DatasourceProtocol"]
            Mapping from source name to instantiated datasource.

        Returns
        -------
        Dict[str, Dict[str, "DatasourceProtocol"]]
            Mapping from split name to the sources assigned to that split.
        """
        pass


class TimeSeriesSplitter(ABC):
    """
    Interface for splitters that partition one time-series payload.
    """

    @abstractmethod
    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split one array into train, validation, and test partitions.

        Parameters
        ----------
        data : numpy.ndarray
            Input array to partition.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Train, validation, and test partitions in that order.
        """
        pass
