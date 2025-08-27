import time
from typing import Any, List, Optional
from numpy import ndarray
from lmetk.data.preprocessing.base import PreProcessorInterface
from lmetk.transforms.base.data_transform import DataTransform
from lmetk.data.data_objects.data import DatasetContainer


import logging

logger = logging.getLogger(__name__)


class PreProcessor(PreProcessorInterface):
    """
    PreProcessor is a data preprocessing class that orchestrates the loading, splitting,
    transformation, and management of datasources.

    See also:
        - `lmetk.data.preprocessing.single_source_loader.SingleSourceLoader`:
                For preprocessing pipelines that operate on a single data source.
        - `lmetk.data.preprocessing.multi_source_loader.MultiSourceLoader`:
                For pipelines that handle multiple data sources simultaneously.
        - `lmetk.data.preprocessing.phmd_loader.PHMDLoader`:
                For specialized loading of PHMD-formatted data sources.

    Attributes:
        mode (str): Specifies the preprocessing mode, either 'per_unit' or 'cross_unit'.
        datasource: The data source object providing access to raw data and metadata.
        transforms: A collection of data transformation objects to be applied sequentially.
        data (Optional[DatasetContainer]): The processed dataset container.
        meta_data_lst (Optional[list[dict]]): List of metadata dictionaries associated with the data.
        _is_preprocessed (bool): Indicates whether the data has been preprocessed.

    Methods:
        __init__(mode, datasource, transforms, **kwargs):
            Initializes the PreProcessor with the specified mode, datasource, and transforms.
        get_meta_data_dict():
            Retrieves metadata from the datasource as a dictionary.
        get_processed_data_dict(return_splits_on_first_level: bool = False) -> dict:
            Returns the processed data as a dictionary, optionally splitting data at the first level.
        fetch_data() -> None:
            Loads and stores data and metadata from the datasource.
        apply_transforms(data: DatasetContainer, transforms) -> List[Any]:
            Applies a sequence of data transformations to the dataset.
        pipeline() -> DatasetContainer:
            Executes the full preprocessing pipeline: loads, splits, fetches, transforms, and returns the processed data.
    """

    def __init__(self, mode, datasource, transforms, **kwargs):
        super().__init__(datasource=datasource, transforms=transforms)

        assert mode in [
            "per_unit",
            "cross_unit",
        ], f"Unsupported mode: {mode}. Supported modes are 'per_unit' and 'cross_unit'."
        self.mode = mode
        self._is_preprocessed = False

        # initialize attributes used later
        self.data: Optional[DatasetContainer] = None
        self.meta_data_lst: Optional[list[dict]] = None

    def get_meta_data_dict(self):
        if hasattr(self.datasource, "get_meta_data") and callable(
            getattr(self.datasource, "get_meta_data", None)
        ):
            meta_data_dict = self.datasource.get_meta_data()
        else:
            meta_data_dict = {}
        return meta_data_dict

    def get_processed_data_dict(
        self, return_splits_on_first_level: bool = False
    ) -> dict[str, ndarray | list[ndarray]]:
        """
        Returns the final processed data.

        The returned dictionary is structured as follows:
        {
            "features": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "target": {
                "train": ndarray | list[ndarray],
                "val": ndarray | list[ndarray],
                "test": ndarray | list[ndarray]
            },
            "other_named_tensor": {
                ...
            }
        }

        If return_splits_on_first_level is True, the function will return the splits at the first level.
        """

        if not self._is_preprocessed or self.data is None:
            raise RuntimeError(
                "Data has not been preprocessed. Please run the pipeline() first."
            )

        if return_splits_on_first_level == True:
            return self.data.to_split_dict()
        else:
            return self.data

    def fetch_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""

        data: DatasetContainer | list[DatasetContainer] = self.datasource.get_data()

        assert isinstance(data, DatasetContainer), (
            f"Expected MultiDataChunk, but got {type(data)}, "
            "make sure 'datasource.get_data()' returns MultiDataChunk"
        )

        if hasattr(self.datasource, "get_meta_data") and callable(
            getattr(self.datasource, "get_meta_data", None)
        ):
            meta_data_lst: list[dict] = self.datasource.get_meta_data()
        else:
            meta_data_lst: list[dict] = [{}]

        if not isinstance(meta_data_lst, list):
            meta_data_lst = [meta_data_lst]

        self.data: DatasetContainer = data
        self.meta_data_lst: list[dict] = meta_data_lst

    def apply_transforms(self, data: DatasetContainer, transforms) -> List[Any]:
        """
        Apply a list of transforms to each data item in data.
        Args:
            data: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        if not all(isinstance(t, DataTransform) for t in transforms.values()):
            raise ValueError(
                "Every element of transforms_lst must be a DataTransform object."
            )

        for transform_name, transform in transforms.items():
            logger.info(
                f"Applying transform: '{transform_name}' "
                f"(Strategy: {transform.strategy.__class__.__name__})"
            )
            start_time = time.time()
            data = transform.forward(data)
            end_time = time.time()
            logger.info(
                f" transform: '{transform.transform_name}' - "
                f"completed in {end_time - start_time:.4f} seconds."
            )

        return data

    def pipeline(self) -> DatasetContainer:
        """Run the preprocessing pipeline."""
        self.datasource.load_data()
        self.datasource.split_data()
        self.fetch_data()  # Loads the data into the pipeline

        assert self.datasource.get_split_mode() == self.mode, (
            f"Expected split mode from preprocessor: '{self.mode}', "
            f"but got dataset split mode: '{self.datasource.get_split_mode()}'"
        )
        self.data = self.apply_transforms(self.data, self.transforms.get_transforms())
        self._is_preprocessed = True

        return self.data
