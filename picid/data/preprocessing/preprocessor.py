import time
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Union
from numpy import ndarray
from picid.data.preprocessing.base import PreProcessorInterface
from picid.transforms.base.data_transform import DataTransform
from picid.data.data_objects.data import DatasetContainer
from picid.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from picid.transforms.base.pipeline import _get_producer_version

from rich import print as rprint
import logging
from picid.data.cache.offline import FileSystemCache
from picid.utils import print_data_dict_structure
from picid.utils.hash_utils import compute_cache_key, ensure_serializable
from picid.utils.rich_output import (
    descriptive_dict_differences_str,
    print_transforms_summary,
    to_descriptive_dict,
)

from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.transforms.base.transform_manager import ConfigTransformManager


logger = logging.getLogger(__name__)


class PreProcessor(PreProcessorInterface):
    """
    PreProcessor is a data preprocessing class that orchestrates the loading, splitting,
    transformation, and management of datasources.

    See also:
        - `picid.data.preprocessing.single_source_loader.SingleSourceLoader`:
                For preprocessing pipelines that operate on a single data source.
        - `picid.data.preprocessing.multi_source_loader.MultiSourceLoader`:
                For pipelines that handle multiple data sources simultaneously.
        - `picid.data.preprocessing.phmd_loader.PHMDLoader`:
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

    def __init__(
        self,
        mode,
        datasource: Union[SingleSourceLoader, MultiSourceLoader],
        transforms: Optional[ConfigTransformManager] = None,
        **kwargs,
    ):

        if transforms is None:
            transforms = ConfigTransformManager(transforms_config=None)
            logger.info("No transforms provided; using empty ConfigTransformManager.")

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
        self._add_datasource_manifest_entry()

    def _add_datasource_manifest_entry(self) -> None:
        """Append a datasource manifest entry to self.data.manifest if present."""
        if self.data is None:
            return
        manifest = getattr(self.data, "manifest", None)
        if manifest is None or not isinstance(manifest, MetadataManifest):
            return
        ds = self.datasource
        payload: dict[str, Any] = {
            "datasource_type": type(ds).__name__,
        }
        # data_name: only a string identifier (e.g. "phme20", "NB14"), not the full metadata dict
        if hasattr(ds, "data_name"):
            val = getattr(ds, "data_name", None)
            if isinstance(val, str):
                payload["data_name"] = val
        if hasattr(ds, "task_mode"):
            payload["task_mode"] = getattr(ds, "task_mode", None)
        if "data_name" not in payload and hasattr(ds, "get_data_name") and callable(getattr(ds, "get_data_name")):
            try:
                name_val = ds.get_data_name()
                if isinstance(name_val, str):
                    payload["data_name"] = name_val
                elif isinstance(name_val, (list, tuple)) and all(isinstance(x, str) for x in name_val):
                    payload["data_name"] = name_val
            except Exception:
                pass
        entry = ManifestEntry(
            schema_version=MANIFEST_SCHEMA_VERSION,
            producer_version=_get_producer_version(),
            category="datasource",
            payload=payload,
            step_id=type(ds).__name__,
            key=None,
            split=None,
        )
        manifest.add(entry)

    def apply_transforms(
        self,
        data: DatasetContainer,
        transforms,
        after_each_transform_callback: Optional[
            Callable[[DatasetContainer, str], None]
        ] = None,
    ) -> List[Any]:
        """
        Apply a list of transforms to each data item in data.
        Args:
            data: List of data items.
            transforms: Dict-like of transform_name -> DataTransform.
            after_each_transform_callback: Optional callback (data, transform_name) called after each transform.
        Returns:
            Transformed data (same type as input).
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
                data, transform_log = transform.forward(data)

                if after_each_transform_callback is not None:
                    after_each_transform_callback(data, transform_name)

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

                if transform_log is not None and transform_log:
                    transform_log = (
                        transform_log["train"]
                        if "train" in transform_log
                        else transform_log["test"]
                    )
                else:
                    transform_log = "No additional transform log."

                summary.append(
                    {
                        "transform_name": transform_name,
                        "time": f" {end_time - start_time:.4f}",
                        "status": "Success",
                        "details": str(transform_log),
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
        save_boundary_caches: bool = True,
    ) -> DatasetContainer:
        """
        Run the preprocessing pipeline using the specific methods of the FileSystemCache.

        Expected behaviour (when cache paths and cache_preprocessed are set):

        1) Load and split data; cache that result (stage "loaded_and_splitted_data").
        2) Run all transforms, then cache the final result (stage "preprocessed").
           If some transform has metadata.cache_point (e.g. a heavy step): run until
           that transform (inclusive), cache that state (stage "boundary"), then run
           the remaining light transforms and cache the final "preprocessed" result.
        3) On a later run, if only config of transforms *after* a cache_point changed:
           full "preprocessed" cache misses (config hash changed), but the "boundary"
           cache (datasource + transforms up to and including the heavy one) is still
           valid. Pipeline restores from that boundary, runs only the light transforms,
           and caches the new final result.

        save_boundary_caches: when True (default), a full run writes boundary caches
        after each cache_point so future runs can resume. Set False to only restore
        from existing boundaries without writing new ones.

        Internal pipeline (when cache paths and cache_preprocessed are set):

        1. Try load final "preprocessed" cache (hash = datasource + full transforms).
           - If HIT: return immediately. No disk writes.

        2. Stage 1: Load/split data (may use "loaded_and_splitted_data" cache).

        3. If any transform has cache_point:
           - Try load "boundary" caches in reverse order (latest boundary first).
           - If a boundary HIT: restore container and manager, run only transforms
             after that boundary, then save only the final "preprocessed" cache and return.
             (No boundary caches are written on this path.)
           - If all boundaries MISS: run all transforms. If save_boundary_caches is True,
             after each transform that is a cache_point we save to stage "boundary"
             (so a later run can resume from there). Then save the final "preprocessed" cache.

        4. If no transform has cache_point: run all transforms, then save
           "preprocessed" once (same as before boundary support).
        """
        datasource_config = ensure_serializable(self.datasource._init_kwargs)
        transforms_config = ensure_serializable(self.transforms.config)

        cache_enabled = bool(
            data_cache_path
            and transform_library_part_path
            and data_library_part_path
            and cache_preprocessed
        )
        if cache_enabled:
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
        else:
            preprocessed_config = combined_paths = expected_hash = None

        # --- Early Exit: Attempt to load the final preprocessed data first ---
        if cache_enabled:
            meta = self.cache.load_metadata(
                data_cache_path, stage="preprocessed", cache_key=expected_hash
            )
            if meta:
                _, stored_hash = meta
                if stored_hash == expected_hash:
                    self.data, saved_objects = self.cache.load_data(
                        data_cache_path,
                        stage="preprocessed",
                        cache_key=expected_hash,
                    )
                    self.transforms = saved_objects["transforms"]
                    self.meta_data = saved_objects["meta_data"]
                    self._is_preprocessed = True
                    logger.info(
                        "✅ Loaded data from Stage 2 (preprocessed) cache. Pipeline complete."
                    )
                    return self.data

        # --- Stage 1: Load or Build Raw Data ---
        def build_data_stage():
            self.datasource.load_data()
            self.datasource.split_data()
            return self.datasource, self.datasource.get_meta_data()

        if data_cache_path and data_library_part_path:
            self.datasource, self.meta_data = self.cache.handle(
                cache_dir=data_cache_path,
                stage="loaded_and_splitted_data",
                build_fn=build_data_stage,
                config=datasource_config,
                library_dir=data_library_part_path,
                extensions=[".py"],
            )
        else:
            self.datasource, self.meta_data = build_data_stage()

        self.fetch_data()

        # --- Stage 2: Apply Transforms (with optional boundary cache restore) ---
        cache_point_names = getattr(
            self.transforms, "get_cache_point_names", lambda: []
        )()
        if cache_enabled and cache_point_names:
            # Try to restore from the latest valid boundary (reverse order)
            for boundary_name in reversed(cache_point_names):
                boundary_config = {
                    "datasource": datasource_config,
                    "transforms": self.transforms.get_config_up_to_and_including(
                        boundary_name
                    ),
                }
                boundary_key = compute_cache_key(
                    config=boundary_config,
                    library_dir=combined_paths,
                    extensions=[".py"],
                )
                meta = self.cache.load_metadata(
                    data_cache_path, stage="boundary", cache_key=boundary_key
                )
                if meta and meta[1] == boundary_key:
                    self.data, saved_objects = self.cache.load_data(
                        data_cache_path,
                        stage="boundary",
                        cache_key=boundary_key,
                    )
                    self.transforms = saved_objects["transforms"]
                    self.meta_data = saved_objects["meta_data"]
                    names_after = self.transforms.get_transform_names_after(
                        boundary_name
                    )
                    if names_after:
                        all_dt = self.transforms.get_data_transforms()
                        transforms_to_run = OrderedDict(
                            (n, all_dt[n]) for n in names_after if n in all_dt
                        )
                        self.data = self.apply_transforms(
                            self.data, transforms_to_run
                        )
                    logger.info(
                        "✅ Restored from boundary cache after %r; ran %d remaining transform(s).",
                        boundary_name,
                        len(names_after),
                    )
                    if cache_enabled:
                        objects_to_cache = {
                            "transforms": self.transforms,
                            "meta_data": self.meta_data,
                        }
                        self.cache.save(
                            cache_dir=data_cache_path,
                            stage="preprocessed",
                            data=self.data,
                            metadata=objects_to_cache,
                            config=preprocessed_config,
                            cache_key=expected_hash,
                        )
                    self._is_preprocessed = True
                    rprint(print_data_dict_structure(self.data))
                    return self.data

        # No boundary hit: run all transforms, optionally saving at each cache point
        if cache_enabled and cache_point_names and save_boundary_caches:

            def save_boundary_callback(data: DatasetContainer, transform_name: str):
                if transform_name not in cache_point_names:
                    return
                boundary_config = {
                    "datasource": datasource_config,
                    "transforms": self.transforms.get_config_up_to_and_including(
                        transform_name
                    ),
                }
                objects = {
                    "transforms": self.transforms,
                    "meta_data": self.meta_data,
                }
                self.cache.save(
                    cache_dir=data_cache_path,
                    stage="boundary",
                    data=data,
                    metadata=objects,
                    config=boundary_config,
                    library_dir=combined_paths,
                    extensions=[".py"],
                )
                logger.info(
                    "Saved boundary cache after transform %r.", transform_name
                )

            self.data = self.apply_transforms(
                self.data,
                self.transforms.get_transforms(),
                after_each_transform_callback=save_boundary_callback,
            )
        else:
            self.data = self.apply_transforms(
                self.data, self.transforms.get_transforms()
            )

        if cache_enabled:
            objects_to_cache = {
                "transforms": self.transforms,
                "meta_data": self.meta_data,
            }
            self.cache.save(
                cache_dir=data_cache_path,
                stage="preprocessed",
                data=self.data,
                metadata=objects_to_cache,
                config=preprocessed_config,
                cache_key=expected_hash,
            )
            logger.info("Applied transforms and saved result to Stage 2 cache.")

        self._is_preprocessed = True
        logger.info("Data dictionary structure after transforms:")
        rprint(print_data_dict_structure(self.data))

        return self.data
