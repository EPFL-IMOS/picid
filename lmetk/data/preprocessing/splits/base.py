from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple

from lmetk.data.datasources import DataSourceLoader


class SourceSplitter(ABC):
    @abstractmethod
    def split_data(self, sources: Dict[str, DataSourceLoader]) -> Dict[str, Dict[str, DataSourceLoader]]:
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
