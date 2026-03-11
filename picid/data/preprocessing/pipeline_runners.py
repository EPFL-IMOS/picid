"""
Pipeline runners for PreProcessor: direct (no cache) and caching (three-tier).

DirectPipelineRunner and CachingPipelineRunner are used by PreProcessor.pipeline()
and are separated here for testability and clearer layout.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from picid.data.data_objects.data import DatasetContainer
from picid.utils.hash_utils import compute_cache_key, ensure_serializable

if TYPE_CHECKING:
    from picid.data.preprocessing.preprocessor import PreProcessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


class DirectPipelineRunner:
    """
    Run the pipeline without any caching.

    Owns: load → split → fetch → transform → return.
    Accepts any TransformSequenceProtocol as transforms.
    """

    def __init__(self, preprocessor: "PreProcessor"):
        self._pp = preprocessor

    def run(self) -> DatasetContainer:
        pp = self._pp
        pp.datasource.load_data()
        pp.datasource.split_data()
        pp.meta_data = pp.datasource.get_meta_data()
        pp.fetch_data()
        pp.data = pp.apply_transforms(pp.data, pp.transforms.get_transforms())
        return pp.data


class CachingPipelineRunner:
    """
    Run the pipeline with a three-tier cache:
        loaded_and_splitted_data  — raw loaded + split container
        boundary                  — state after a designated heavy transform
        preprocessed              — fully transformed final result

    Works with any object satisfying TransformSequenceProtocol, so both
    ConfigTransformManager (framework) and TransformPipeline (library) are
    supported without any changes here.

    Cache key derivation
    --------------------
    The preprocessed key is computed from:
        - datasource._init_kwargs  (serialised)
        - transforms.config        (serialised; stable for same pipeline definition)
        - library source files in data_library_part_path + transform_library_part_path

    For ConfigTransformManager, transforms.config is the OmegaConf DictConfig.
    For TransformPipeline, transforms.config is a plain dict built from transform
    class names and metadata dicts.  Both are compatible with compute_cache_key.

    Parameters
    ----------
    preprocessor : PreProcessor
    data_cache_path : str
    data_library_part_path : str
    transform_library_part_path : str
    save_boundary_caches : bool
    """

    def __init__(
        self,
        preprocessor: "PreProcessor",
        data_cache_path: str,
        data_library_part_path: str,
        transform_library_part_path: str,
        save_boundary_caches: bool,
    ):
        self._pp = preprocessor
        self._cache_dir = data_cache_path
        self._data_lib = data_library_part_path
        self._transform_lib = transform_library_part_path
        self._save_boundaries = save_boundary_caches
        self._combined_paths = [data_library_part_path, transform_library_part_path]

        pp = preprocessor
        self._datasource_config = ensure_serializable(pp.datasource._init_kwargs)
        self._transforms_config = ensure_serializable(pp.transforms.config)
        self._preprocessed_config = {
            "datasource": self._datasource_config,
            "transforms": self._transforms_config,
        }
        self._preprocessed_key = compute_cache_key(
            config=self._preprocessed_config,
            library_dir=self._combined_paths,
            extensions=[".py"],
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> DatasetContainer:
        # 1. Try final preprocessed cache first.
        result = self._try_load_preprocessed()
        if result is not None:
            return result

        # 2. Load and split (uses loaded_and_splitted_data cache).
        self._load_and_split()

        # 3. Try to restore from a boundary cache.
        result = self._try_restore_from_boundary()
        if result is not None:
            return result

        # 4. Full run: all transforms, optionally saving at each cache point.
        self._run_all_transforms()

        # 5. Save final preprocessed cache.
        self._save_preprocessed()

        return self._pp.data

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _try_load_preprocessed(self) -> Optional[DatasetContainer]:
        """Return cached final result if the hash matches, else None."""
        pp = self._pp
        meta = pp.cache.load_metadata(
            self._cache_dir, stage="preprocessed", cache_key=self._preprocessed_key
        )
        if meta and meta[1] == self._preprocessed_key:
            pp.data, saved = pp.cache.load_data(
                self._cache_dir,
                stage="preprocessed",
                cache_key=self._preprocessed_key,
            )
            pp.transforms = saved["transforms"]
            pp.meta_data = saved["meta_data"]
            logger.info("Loaded data from preprocessed cache. Pipeline complete.")
            return pp.data
        return None

    def _load_and_split(self) -> None:
        """Load and split via cache or direct build."""
        pp = self._pp

        def _build():
            pp.datasource.load_data()
            pp.datasource.split_data()
            return pp.datasource, pp.datasource.get_meta_data()

        pp.datasource, pp.meta_data = pp.cache.handle(
            cache_dir=self._cache_dir,
            stage="loaded_and_splitted_data",
            build_fn=_build,
            config=self._datasource_config,
            library_dir=self._data_lib,
            extensions=[".py"],
        )
        pp.fetch_data()

    def _try_restore_from_boundary(self) -> Optional[DatasetContainer]:
        """
        Try to restore from the latest valid boundary cache (reverse order).

        Returns the final DatasetContainer if a boundary was restored and the
        remaining transforms were applied, or None if no boundary matched.
        """
        pp = self._pp
        cache_point_names = pp.transforms.get_cache_point_names()
        if not cache_point_names:
            return None

        for boundary_name in reversed(cache_point_names):
            boundary_config = {
                "datasource": self._datasource_config,
                "transforms": pp.transforms.get_config_up_to_and_including(
                    boundary_name
                ),
            }
            boundary_key = compute_cache_key(
                config=boundary_config,
                library_dir=self._combined_paths,
                extensions=[".py"],
            )
            meta = pp.cache.load_metadata(
                self._cache_dir, stage="boundary", cache_key=boundary_key
            )
            if not (meta and meta[1] == boundary_key):
                continue

            pp.data, saved = pp.cache.load_data(
                self._cache_dir, stage="boundary", cache_key=boundary_key
            )
            pp.transforms = saved["transforms"]
            pp.meta_data = saved["meta_data"]

            names_after = pp.transforms.get_transform_names_after(boundary_name)
            if names_after:
                all_dt = pp.transforms.get_transforms()
                tail = OrderedDict((n, all_dt[n]) for n in names_after if n in all_dt)
                pp.data = pp.apply_transforms(pp.data, tail)

            logger.info(
                "Restored from boundary cache after %r; ran %d remaining transform(s).",
                boundary_name,
                len(names_after),
            )
            self._save_preprocessed()
            return pp.data

        return None

    def _run_all_transforms(self) -> None:
        """
        Apply all transforms, optionally saving a boundary cache after each
        transform marked as a cache point.
        """
        pp = self._pp
        cache_point_names = pp.transforms.get_cache_point_names()

        if cache_point_names and self._save_boundaries:

            def _boundary_callback(data: DatasetContainer, transform_name: str):
                if transform_name not in cache_point_names:
                    return
                boundary_config = {
                    "datasource": self._datasource_config,
                    "transforms": pp.transforms.get_config_up_to_and_including(
                        transform_name
                    ),
                }
                pp.cache.save(
                    cache_dir=self._cache_dir,
                    stage="boundary",
                    data=data,
                    metadata={"transforms": pp.transforms, "meta_data": pp.meta_data},
                    config=boundary_config,
                    library_dir=self._combined_paths,
                    extensions=[".py"],
                )
                logger.info("Saved boundary cache after transform %r.", transform_name)

            pp.data = pp.apply_transforms(
                pp.data,
                pp.transforms.get_transforms(),
                after_each_transform_callback=_boundary_callback,
            )
        else:
            pp.data = pp.apply_transforms(pp.data, pp.transforms.get_transforms())

    def _save_preprocessed(self) -> None:
        """Save the fully transformed container to the preprocessed cache."""
        pp = self._pp
        pp.cache.save(
            cache_dir=self._cache_dir,
            stage="preprocessed",
            data=pp.data,
            metadata={"transforms": pp.transforms, "meta_data": pp.meta_data},
            config=self._preprocessed_config,
            cache_key=self._preprocessed_key,
        )
        logger.info("Saved result to preprocessed cache.")
