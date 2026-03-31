"""
Pipeline runners for PreProcessor: direct (no cache) and caching (three-tier).

DirectPipelineRunner and CachingPipelineRunner are used by PreProcessor.pipeline()
and are separated here for testability and clearer layout.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

from picid.data.data_objects import DatasetContainer
from picid.data.datasources.base.exceptions import (
    DatasourceContractError,
    DatasourceError,
)
from picid.exceptions import build_preprocessing_datasource_error
from picid.utils.hash_utils import compute_cache_key, ensure_serializable

if TYPE_CHECKING:
    from picid.data.preprocessing.preprocessor import PreProcessor

logger = logging.getLogger(__name__)


def _resolve_datasource_cache_fingerprint(datasource: object) -> dict:
    """
    Return the datasource fingerprint used for preprocessing cache keys.

    Parameters
    ----------
    datasource : object
        Datasource instance expected to expose ``get_cache_fingerprint()``.

    Returns
    -------
    dict
        Serialisable datasource fingerprint for cache key derivation.

    Raises
    ------
    DatasourceContractError
        If the datasource does not implement ``get_cache_fingerprint()``.
    """
    getter = getattr(datasource, "get_cache_fingerprint", None)
    if not callable(getter):
        raise DatasourceContractError(
            "Datasource must implement get_cache_fingerprint() for cached preprocessing."
        )
    return getter()


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------


class DirectPipelineRunner:
    """
    Run preprocessing end to end without caching.

    This path loads, splits, materializes, and transforms the dataset without
    persisting intermediate cache artifacts.

    Parameters
    ----------
    preprocessor : PreProcessor
        Owning preprocessor object.
    """

    def __init__(self, preprocessor: "PreProcessor"):
        self._pp = preprocessor

    def run(self) -> DatasetContainer:
        """
        Run the non-cached preprocessing pipeline end to end.

        Returns
        -------
        DatasetContainer
            Fully transformed dataset container.

        Raises
        ------
        PreprocessingDatasourceError
            If a datasource-stage call fails with a typed datasource exception.
        """
        pp = self._pp
        pp._run_datasource_step("load_data", pp.datasource.load_data)
        pp._run_datasource_step("split_data", pp.datasource.split_data)
        pp.meta_data = pp._run_datasource_step(
            "get_meta_data", pp.datasource.get_meta_data
        )
        pp.fetch_data()
        pp.data = pp.apply_transforms(pp.data, pp.transforms.get_transforms())
        return pp.data


class CachingPipelineRunner:
    """
    Run preprocessing with the three-tier caching strategy.

    The runner first tries the final cache, then the loaded/split cache, then
    boundary caches produced after cache-point transforms.

    Parameters
    ----------
    preprocessor : PreProcessor
        Owning preprocessor object.
    data_cache_path : str
        Root directory for cache artifacts.
    data_library_part_path : str
        Path used in cache key derivation for datasource code.
    transform_library_part_path : str
        Path used in cache key derivation for transform code.
    save_boundary_caches : bool
        Whether boundary caches should be persisted.

    Notes
    -----
    Cache stages:

    - ``loaded_and_splitted_data`` stores the raw loaded and split container.
    - ``boundary`` stores state after designated heavy transforms.
    - ``preprocessed`` stores the fully transformed final result.

    The final cache key is derived from three inputs:

    - ``datasource.get_cache_fingerprint()``, serialized.
    - ``transforms.config``, serialized in a stable form.
    - Python source files under ``data_library_part_path`` and
      ``transform_library_part_path``.

    ``ConfigTransformManager`` and ``TransformPipeline`` expose different
    concrete config types, but both are normalized into serializable values and
    hashed the same way here.
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
        try:
            datasource_fingerprint = _resolve_datasource_cache_fingerprint(
                pp.datasource
            )
        except DatasourceError as exc:
            raise build_preprocessing_datasource_error(
                "get_cache_fingerprint", pp.datasource, exc
            ) from exc
        self._datasource_config = ensure_serializable(datasource_fingerprint)
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
        #    Returns the tail OrderedDict of remaining transforms if a boundary
        #    was hit (possibly empty), or None if no boundary matched.
        tail = self._try_restore_from_boundary()

        # 4. Pre-write preprocessed tombstone before any transforms run.
        self._pp.cache.write_meta(
            cache_dir=self._cache_dir,
            stage="preprocessed",
            config=self._preprocessed_config,
            cache_key=self._preprocessed_key,
        )

        # 5. Run transforms: full chain, or only the tail after a boundary restore.
        #    Passing `tail` lets _run_all_transforms handle any new cache points
        #    that may appear among the remaining transforms.
        self._run_all_transforms(transforms=tail)

        # 6. Save final preprocessed cache.
        self._save_preprocessed()

        return self._pp.data

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _try_load_preprocessed(self) -> Optional[DatasetContainer]:
        """
        Return cached final result if the hash matches, else ``None``.

        Returns
        -------
        DatasetContainer | None
            Cached final result when available.
        """
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
        """
        Load and split the datasource via cache or direct build.

        Returns
        -------
        None
            The method updates the preprocessor datasource, metadata, and data
            in place.

        Raises
        ------
        PreprocessingDatasourceError
            If the datasource fails during cache-fingerprint resolution,
            loading, splitting, or metadata retrieval with a typed datasource
            exception.
        """
        pp = self._pp

        def _build():
            pp._run_datasource_step("load_data", pp.datasource.load_data)
            pp._run_datasource_step("split_data", pp.datasource.split_data)
            meta_data = pp._run_datasource_step(
                "get_meta_data", pp.datasource.get_meta_data
            )
            return pp.datasource, meta_data

        pp.datasource, pp.meta_data = pp.cache.handle(
            cache_dir=self._cache_dir,
            stage="loaded_and_splitted_data",
            build_fn=_build,
            config=self._datasource_config,
            library_dir=self._data_lib,
            extensions=[".py"],
        )
        pp.fetch_data()

    def _try_restore_from_boundary(self) -> Optional[OrderedDict]:
        """
        Try to restore from the latest valid boundary cache.

        Returns an OrderedDict of the remaining (not-yet-applied) transforms if a
        boundary was successfully restored, or None if no boundary matched.  The
        returned dict may be empty when the boundary was the last transform in the
        current chain.

        The caller is responsible for running the returned tail transforms via
        _run_all_transforms so that any new cache points among them are handled
        correctly.
        Returns
        -------
        DatasetContainer | None
            Restored and completed dataset container, or ``None`` when no
            boundary cache matched.
        """
        pp = self._pp
        cache_point_names = pp.transforms.get_cache_point_names()
        if not cache_point_names:
            return None

        full_transforms = pp.transforms

        for boundary_name in reversed(cache_point_names):
            boundary_config = {
                "datasource": self._datasource_config,
                "transforms": full_transforms.get_config_up_to_and_including(
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
            pp.meta_data = saved["meta_data"]

            # Restore fitted transform instances from the saved boundary into the
            # current manager.  Only transforms that exist in both the saved manager
            # and the current chain are updated; transforms added after the boundary
            # was originally written are left unfitted and will be run via the tail.
            saved_dts = saved["transforms"].get_transforms()
            current_dts = full_transforms.get_transforms()
            for name, saved_dt in saved_dts.items():
                if name in current_dts:
                    current_dts[name].transform_instance = saved_dt.transform_instance

            names_after = full_transforms.get_transform_names_after(boundary_name)
            tail = OrderedDict(
                (n, current_dts[n]) for n in names_after if n in current_dts
            )

            logger.info(
                "Restored from boundary cache after %r; %d remaining transform(s) to run.",
                boundary_name,
                len(tail),
            )
            return tail

        return None

    def _run_all_transforms(self, transforms: Optional[OrderedDict] = None) -> None:
        """
        Apply transforms, optionally saving a boundary cache after each cache point.

        Parameters
        ----------
        transforms:
            OrderedDict of DataTransform objects to apply.  When None (the default),
            the full transform chain from pp.transforms is used.  Pass the tail dict
            returned by _try_restore_from_boundary to run only the remaining transforms
            after a boundary restore; boundary callbacks are still applied correctly
            for any cache points present in the tail.
        """
        pp = self._pp
        if transforms is None:
            transforms = pp.transforms.get_transforms()

        cache_point_names = pp.transforms.get_cache_point_names()
        # Only consider cache points that are actually in the transforms to be run.
        active_cache_points = [n for n in cache_point_names if n in transforms]

        if active_cache_points and self._save_boundaries:
            # Pre-write boundary tombstones for active cache points before any
            # transform runs, so a crash leaves a meta.json on disk.
            for boundary_name in active_cache_points:
                boundary_config = {
                    "datasource": self._datasource_config,
                    "transforms": pp.transforms.get_config_up_to_and_including(
                        boundary_name
                    ),
                }
                pp.cache.write_meta(
                    cache_dir=self._cache_dir,
                    stage="boundary",
                    config=boundary_config,
                    library_dir=self._combined_paths,
                    extensions=[".py"],
                )

            def _boundary_callback(data: DatasetContainer, transform_name: str):
                if transform_name not in active_cache_points:
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
                transforms,
                after_each_transform_callback=_boundary_callback,
            )
        else:
            pp.data = pp.apply_transforms(pp.data, transforms)

    def _save_preprocessed(self) -> None:
        """
        Save the fully transformed container to the preprocessed cache.
        """
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
