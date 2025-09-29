import time
from typing import Any, List, Optional
from numpy import ndarray
from lmetk.data.preprocessing.base import PreProcessorInterface
from lmetk.transforms.base.data_transform import DataTransform
from lmetk.data.data_objects.data import DatasetContainer


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
        for transform_name, transform in transforms.items():
            logger.info(
                f"Applying transform: '{transform_name}' "
                f"(Strategy: {transform.strategy.__class__.__name__})"
            )
            pre_descr_dict = to_descriptive_dict(data.to_split_dict()["train"])
            start_time = time.time()
            data, details = transform.forward(data)

            end_time = time.time()
            logger.info(
                f" transform: '{transform.transform_name}' - "
                f"completed in {end_time - start_time:.4f} seconds."
            )
            post_descr_dict = to_descriptive_dict(data.to_split_dict()["train"])

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
                }
            )

        print_transforms_summary(summary)
        return data

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
        Run the preprocessing pipeline with optional caching.

        This method orchestrates the entire data preprocessing workflow, including
        loading data, applying transformations, and leveraging a two-stage cache
        to save intermediate results.

        - Stage 1 Cache: Caches the data after it has been loaded and split.
        - Stage 2 Cache: Caches the data after all transformations have been applied.
        This cache also saves the state of the fitted transforms and metadata.
        """

        # -----------------
        # Stage 2: Preprocessed cache check first (skip stage 1 if it exists)
        # -----------------
        datasource_config = ensure_serializable(self.datasource._init_kwargs)
        datasource_transforms_cfg = {
            "datasource": datasource_config,
            "transforms": ensure_serializable(self.transforms.config),
        }

        preprocessed_data_loaded = False
        if data_cache_path and transform_library_part_path and cache_preprocessed:
            expected_hash = compute_cache_key(
                config=datasource_transforms_cfg,
                library_dir=transform_library_part_path,
                extensions=[".py"],
            )

            preprocessed_meta = self.cache.load_metadata(
                data_cache_path,
                stage="preprocessed",
                cache_key=expected_hash,
            )

            if preprocessed_meta:
                _, stored_hash = preprocessed_meta
                if stored_hash == expected_hash:
                    # Cache hit -> load data and the container with our objects
                    loaded_data, saved_objects = self.cache.load_data(
                        data_cache_path, stage="preprocessed", cache_key=expected_hash
                    )
                    self.data = loaded_data

                    # Unpack the dictionary from the cache
                    self.transforms = saved_objects["transforms"]
                    self.meta_data = saved_objects["meta_data"]
                    preprocessed_data_loaded = True

        # -----------------
        # Stage 1: Load + Split + Fetch
        # -----------------
        if not preprocessed_data_loaded:
            if data_cache_path:
                if not data_library_part_path:
                    raise ValueError(
                        "data_library_part_path must be provided if caching loaded data"
                    )

                def build_data_stage():
                    self.datasource.load_data()
                    self.datasource.split_data()
                    return self.datasource, self.datasource.get_meta_data()

                self.datasource, self.meta_data = self.cache.handle(
                    cache_dir=data_cache_path,
                    stage="loaded_and_splitted_data",
                    build_fn=build_data_stage,
                    config=datasource_config,
                    library_dir=data_library_part_path,
                    extensions=[".py"],
                )
            else:
                self.datasource.load_data()
                self.datasource.split_data()

            self.fetch_data()

            assert self.datasource.get_split_mode() == self.mode, (
                f"Expected split mode from preprocessor: '{self.mode}', "
                f"but got dataset split mode: '{self.datasource.get_split_mode()}'"
            )

            logger.info("Data dictionary structure before transforms:")
            print_data_dict_structure(self.data)

        # -----------------
        # Stage 2: Transforms
        # -----------------
        if not preprocessed_data_loaded:
            if data_cache_path and transform_library_part_path and cache_preprocessed:

                def build_preprocess_stage():
                    processed_data = self.apply_transforms(
                        self.data, self.transforms.get_transforms()
                    )

                    # Create a dictionary to hold both objects to be saved in metadata.pkl
                    objects_to_cache = {
                        "transforms": self.transforms,
                        "meta_data": self.meta_data,
                    }
                    return (processed_data, objects_to_cache)

                # The cache handles the `objects_to_cache` dict, saving it to metadata.pkl
                cached_data = self.cache.handle(
                    cache_dir=data_cache_path,
                    stage="preprocessed",
                    build_fn=build_preprocess_stage,
                    config=datasource_transforms_cfg,
                    library_dir=transform_library_part_path,
                    extensions=[".py"],
                )

                # Unpack the results to keep instance attributes consistent
                self.data, saved_objects = cached_data
                self.transforms = saved_objects["transforms"]
                self.meta_data = saved_objects["meta_data"]

            else:
                self.data = self.apply_transforms(
                    self.data, self.transforms.get_transforms()
                )

        self._is_preprocessed = True

        logger.info("Data dictionary structure after transforms:")
        print_data_dict_structure(self.data)

        return self.data

    def get_chached_transform_manager(self):
        return self.transforms
