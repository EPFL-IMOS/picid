from abc import ABC, abstractmethod
from omegaconf import DictConfig


class TimeSeriesDatasetInterface(ABC):
    @abstractmethod
    def __init__(self, load_arguments, data_splitter, scaler):
        pass

    @abstractmethod
    def get_data(self):
        """Return splitted data and timestamps."""
        pass

    @abstractmethod
    def get_scaler(self):
        """Return the fitted scaler."""
        pass

    @abstractmethod
    def _pipeline(self):
        """Load, split, and transform the data."""
        pass

    @abstractmethod
    def _load_data(self):
        """Load the raw dataset and timestamp information."""
        pass

    @abstractmethod
    def _generate_splits(self):
        """Split the data and timestamps into train/val/test sets."""
        pass

    @abstractmethod
    def _apply_transformations(self):
        """Apply any preprocessing, such as scaling."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class AbstractTimeSeriesDataSource(TimeSeriesDatasetInterface):
    def __init__(self, load_arguments, data_splitter, scaler):

        self.load_arguments: DictConfig = load_arguments
        self.data_splitter = data_splitter
        self.scaler = scaler
        # self._pipeline()

    def get_data(self):
        return self.splitted_data, self.splitted_timestamps

    def get_scaler(self):
        return self.scaler

    def _pipeline(self):
        self.data, self.timestamps, mask = self._load_data()
        self.splitted_data, self.splitted_timestamps = self._generate_splits()
        self._apply_transformations()

    @abstractmethod
    def _load_data(self):
        pass

    def _generate_splits(self):
        # Split the data (e.g., train/val/test sequences)
        train_data, val_data, test_data = self.data_splitter.split_data(self.data)
        data = {"train": train_data, "val": val_data, "test": test_data}

        train_timestamps, val_timestamps, test_timestamps = self.data_splitter.split_data(self.timestamps)
        timestamps = {"train": train_timestamps, "val": val_timestamps, "test": test_timestamps}
        return data, timestamps

    def _apply_transformations(self):
        # Optionally scale the data
        if self.load_arguments.get("scale", True):
            
            self.scaler.fit(self.splitted_data["train"])
            
            for split_name in ["train", "val", "test"]:
                if split_name in self.splitted_data:
                    self.splitted_data[split_name] = self.scaler.transform(self.splitted_data[split_name])
    
    def __repr__(self) -> str:
        return f"loader_arguments: {self.load_arguments}, " \
                f"data_splitter: {self.data_splitter}, " \
                f"scaler: {self.scaler}"