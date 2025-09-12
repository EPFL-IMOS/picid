from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray

from lmetk.data.datasources import SingleSourceLoader
from lmetk.transforms.base.transform_manager import ConfigTransformManager


class PreProcessorInterface(ABC):
    """Preprocessor of data."""

    def __init__(
        self,
        datasource: Union[SingleSourceLoader, SingleSourceLoader],
        transforms: ConfigTransformManager,
        **kwargs,
    ):
        self.datasource = datasource
        self.transforms = transforms

    @abstractmethod
    def get_processed_data_dict(
        self,
    ) -> dict[str, dict[str, ndarray | list[ndarray]]]:
        """
        Returns the final processed data
        the dictionary should be structured as follows:
        - features:
            - train: ndarray | list[ndarray]
            - val: ndarray | list[ndarray]
            - test: ndarray | list[ndarray]
        - target:
            - train: ndarray | list[ndarray]
            - val: ndarray | list[ndarray]
            - test: ndarray | list[ndarray]
        - other_named_tensor:
            - ...
        """
        raise NotImplementedError("get_processed_data method not implemented.")

    @abstractmethod
    def get_meta_data_dict(self):
        raise NotImplementedError("get_meta_data_dict method not implemented.")

    @abstractmethod
    def fetch_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        raise NotImplementedError("prepare_data method not implemented.")

    @abstractmethod
    def apply_transforms(self) -> None:
        """Apply the transforms to the datasource."""
        raise NotImplementedError("apply_transforms method not implemented.")

    @abstractmethod
    def pipeline(self):
        """Run the preprocessing pipeline."""
        raise NotImplementedError("pipeline method not implemented.")
