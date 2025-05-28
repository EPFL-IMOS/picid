from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class TimeSeriesSplitter(ABC):
    @abstractmethod
    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
