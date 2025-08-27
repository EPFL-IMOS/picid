import time
from typing import Any, List, Optional

from numpy import ndarray

from lmetk.data.preprocessing.base import PreProcessorInterface
from lmetk.transforms.base.data_transform import DataTransform


class PreProcessor(PreProcessorInterface):
    """Preprocessor of data."""

    def __init__(self, datasource, transforms, logger=None, **kwargs):
        super().__init__(datasource=datasource, transforms=transforms, logger=logger)

        self.data_lst: Optional[List[Any]] = None
        self._is_preprocessed = False

    def get_processed_data(self) -> dict[str, ndarray | list[ndarray]]:
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
        """
        if not self._is_preprocessed or self.data_lst is None:
            raise RuntimeError(
                "Data has not been preprocessed. Please run the pipeline() first."
            )

        if isinstance(self.data_lst, list) and len(self.data_lst) == 1:
            return self.data_lst[0]

        else:
            raise ValueError("Processed data is in an unexpected format.")

    # TODO: that is needed at least for evaluator, extend it for multisource maybe
    def get_transform_names(
        self,
    ):
        return self.source_transform_names_lst

    def fetch_data(self) -> None:
        """Prepare data from the datasource and store as instance variables."""
        data_lst = self.datasource.get_data()

        # Ensure all outputs are lists for consistent processing
        if not isinstance(data_lst, list):
            data_lst = [data_lst]
        self.data_lst = data_lst

    def apply_transforms(self, data_lst: List[Any], transforms) -> List[Any]:
        """
        Apply a list of transforms to each data item in data_lst.
        Args:
            data_lst: List of data items.
            transforms_lst: List of lists of DataTransform objects, one list per data item.
        Returns:
            List of transformed data items.
        """
        if not all(isinstance(t, DataTransform) for t in transforms.values()):
            raise ValueError(
                "Every element of transforms_lst must be a DataTransform object."
            )

        transformed_data_lst = []
        for current_data in data_lst:
            for transform_name, transform in transforms.items():
                # Removed detailed print statements for each transform application.
                if self.logger:
                    self.logger.info(
                        f"Applying transform: '{transform_name}' (Strategy: {transform.strategy.__class__.__name__})"
                    )
                start_time = time.time()
                current_data = transform.forward(current_data)
                end_time = time.time()

                if self.logger:
                    self.logger.info(
                        f" transform: '{transform.transform_name}' - completed in {end_time - start_time:.4f} seconds."
                    )

            transformed_data_lst.append(current_data)
        return transformed_data_lst

    def pipeline(self):
        """Run the preprocessing pipeline."""
        # Removed "Starting preprocessing pipeline..." print.
        self.fetch_data()
        # if multisource_data_splitter do not exist, then we work with separate sources
        assert (
            self.datasource.get_split_mode() == "per_unit"
        ), "That is a singlesource preprocessing hence datasource {self.datasource.get_split_mode()} should have 'per_unit' split mode"
        self.data_lst = self.apply_transforms(
            self.data_lst, self.transforms.get_transforms()
        )
        self._is_preprocessed = True

        return self.data_lst


def key_exists_anywhere(data, target_key):
    # Check if object has dict-like methods
    if hasattr(data, "keys") and hasattr(data, "values") and hasattr(data, "items"):
        # First check if the target key exists at this level
        if target_key in data:
            return True, data[target_key]

        # Then recursively search in all values
        for value in data.values():
            found, val = key_exists_anywhere(value, target_key)
            if found:
                return found, val

    elif isinstance(data, (list, tuple)):
        for item in data:
            found, val = key_exists_anywhere(item, target_key)
            if found:
                return found, val

    return False, None
