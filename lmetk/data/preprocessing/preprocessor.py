import time
from typing import Any, List, Optional, Union
from numpy import ndarray
from lmetk.data.preprocessing.base import PreProcessorInterface
from lmetk.transforms.base.data_transform import DataTransform
from lmetk.data.data_objects.data import DatasetContainer

from rich import print as rprint
import logging
from lmetk.data.cache.offline import FileSystemCache
from lmetk.utils import print_data_dict_structure
from lmetk.utils.hash_utils import compute_cache_key, ensure_serializable
from lmetk.utils.rich_output import (
    descriptive_dict_differences_str,
    print_transforms_summary,
    to_descriptive_dict,
)


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
            Executes the full preprocessing pipeline: loads, splits, fetches,
            transforms, and returns the processed data.
    """

    def __init__(self, mode, datasource, transforms, **kwargs):
        super().__init__(datasource=datasource, transforms=transforms)

        assert mode in [
            "per_unit",
            "cross_unit",
        ], (
            f"Unsupported mode: {mode}. Supported modes are 'per_unit'"
            " and 'cross_unit'."
        )

        self.mode = mode
        self._is_preprocessed = False
        self.data: Optional[DatasetContainer] = None
        self.meta_data: dict = {}
        self.cache = FileSystemCache()

    def get_meta_data_dict(self):
        return self.meta_data

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

        self.data: DatasetContainer = data

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

        summary = []

        try:
            for transform_name, transform in transforms.items():
                logger.info(
                    f"Applying transform: '{transform_name}' "
                    f"(Strategy: {transform.strategy.__class__.__name__})"
                )
                pre_descr_dict = to_descriptive_dict(
                    data.to_split_dict()["train"], calculate_stat=False
                )
                start_time = time.time()
                data, details = transform.forward(data)

                end_time = time.time()
                logger.info(
                    f" transform: '{transform.transform_name}' - "
                    f"completed in {end_time - start_time:.4f} seconds."
                )
                post_descr_dict = to_descriptive_dict(
                    data.to_split_dict()["train"], calculate_stat=False
                )

                changes = descriptive_dict_differences_str(
                    pre_descr_dict, post_descr_dict, mode="changed"
                )
                added = descriptive_dict_differences_str(
                    pre_descr_dict, post_descr_dict, mode="added"
                )
                removed = descriptive_dict_differences_str(
                    pre_descr_dict, post_descr_dict, mode="removed"
                )
                summary.append(
                    {
                        "transform_name": transform_name,
                        "time": f" {end_time - start_time:.4f}",
                        "status": "Success",
                        "details": details,
                        "changes": str(changes),
                        "added": str(added),
                        "removed": str(removed),
                        "inputs": str(pre_descr_dict),
                    }
                )
        finally:
            print_transforms_summary(summary)
        return data

    def get_chached_transform_manager(self):
        return self.transforms

    # configs of the dataset
    # configs of the transforms

    # 1) cache after the loading -- do not need to load long time
    # 2) cache after the preprocessing -- do not need to preprocess
    # 3) configs of the dataset + but different transforms then laod from the 1) and overwrite 2)

    def pipeline(
        self,
        data_cache_path: str = None,
        data_library_part_path: str = None,
        transform_library_part_path: str = None,
        cache_preprocessed: bool = False,
    ) -> DatasetContainer:
        """
        Run the preprocessing pipeline using the specific methods of the FileSystemCache.
        """
        datasource_config = ensure_serializable(self.datasource._init_kwargs)
        transforms_config = ensure_serializable(self.transforms.config)

        # --- Early Exit: Attempt to load the final preprocessed data first ---
        if (
            data_cache_path
            and transform_library_part_path
            and data_library_part_path
            and cache_preprocessed
        ):
            preprocessed_config = {
                "datasource": datasource_config,
                "transforms": transforms_config,
            }
            combined_paths = [data_library_part_path, transform_library_part_path]

            expected_hash = compute_cache_key(
                config=preprocessed_config,
                library_dir=combined_paths,
                extensions=[".py"],
            )

            # 1. Check for a valid cache using .load_metadata() without loading heavy data
            meta = self.cache.load_metadata(
                data_cache_path, stage="preprocessed", cache_key=expected_hash
            )

            if meta:
                _, stored_hash = meta
                if stored_hash == expected_hash:
                    # 2. If metadata is valid, load the full data using .load_data()
                    self.data, saved_objects = self.cache.load_data(
                        data_cache_path, stage="preprocessed", cache_key=expected_hash
                    )
                    self.transforms = saved_objects["transforms"]
                    self.meta_data = saved_objects["meta_data"]
                    self._is_preprocessed = True
                    logger.info(
                        "✅ Loaded data from Stage 2 (preprocessed) cache. Pipeline complete."
                    )
                    return self.data  # ✅ Early return on cache hit

        # --- Stage 1: Load or Build Raw Data using the .handle() method ---
        # This part only runs if the preprocessed cache was missed.
        def build_data_stage():
            self.datasource.load_data()
            self.datasource.split_data()
            # The .handle() method expects a tuple of (data, metadata) from the build function
            return self.datasource, self.datasource.get_meta_data()

        if data_cache_path and data_library_part_path:
            # Use the convenient .handle() method which encapsulates the entire
            # "load or build and save" logic for Stage 1.
            self.datasource, self.meta_data = self.cache.handle(
                cache_dir=data_cache_path,
                stage="loaded_and_splitted_data",
                build_fn=build_data_stage,
                config=datasource_config,
                library_dir=data_library_part_path,
                extensions=[".py"],
            )
        else:  # Fallback if caching is disabled for this stage
            self.datasource, self.meta_data = build_data_stage()

        self.fetch_data()

        # --- Stage 2: Apply Transforms and Save Result ---
        # This part always runs if we haven't returned early.
        self.data = self.apply_transforms(self.data, self.transforms.get_transforms())

        # Save the result to the Stage 2 cache if enabled
        if (
            data_cache_path
            and transform_library_part_path
            and data_library_part_path
            and cache_preprocessed
        ):
            objects_to_cache = {
                "transforms": self.transforms,
                "meta_data": self.meta_data,
            }

            # Use the specific .save() method with all its required arguments
            self.cache.save(
                cache_dir=data_cache_path,
                stage="preprocessed",
                data=self.data,
                metadata=objects_to_cache,
                config=preprocessed_config,
                cache_key=expected_hash,  # Pass the pre-computed key
            )
            logger.info("Applied transforms and saved result to Stage 2 cache.")

        self._is_preprocessed = True
        logger.info("Data dictionary structure after transforms:")
        # rprint(print_data_dict_structure(self.data)) # Assuming rprint is defined elsewhere

        return self.data
