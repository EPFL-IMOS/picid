from abc import ABC, abstractmethod

class BaseCache(ABC):
    """Abstract base class for all cache backends."""

    @abstractmethod
    def load(self, name: str):
        pass

    @abstractmethod
    def save(self, name: str, data, config: dict = None):
        pass

    @abstractmethod
    def handle(self, name: str, build_fn, config: dict = None):
        """Load if valid, otherwise build and save."""
        pass

