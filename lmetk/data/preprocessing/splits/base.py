from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

from lmetk.data.datasources.base.single_source_loader import SingleSourceLoader


class SourceSplitter(ABC):
    @abstractmethod
    def split_data(
        self, sources: Dict[str, SingleSourceLoader]
    ) -> Dict[str, Dict[str, SingleSourceLoader]]:
        """
        Split the data by source.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            Dictionary containing data arrays for each source.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing split data arrays for each source.
        """
        pass


class TimeSeriesSplitter(ABC):
    @abstractmethod
    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
