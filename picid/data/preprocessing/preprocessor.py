"""
preprocessor.py
===============

PreProcessor orchestrates loading, splitting, transformation, and caching of
datasets for both framework and library users.

Framework users (run.py)
------------------------
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor("per_unit", datasource, transforms_manager)
    data = preprocessor.pipeline(cache_paths=..., cache_preprocessed=True)

Library users
-------------
    from picid.transforms.base.transform_pipeline import TransformPipeline

    pipeline = TransformPipeline([
        DataTransform("normalize", scaler,  {"apply_to": "features", "fit_on": "train"}),
        DataTransform("window",    windower, {"apply_to": "features", "cache_point": True}),
        DataTransform("scale_y",   scaler2, {"apply_to": "target",   "fit_on": "train"}),
    ])

    # Without caching:
    data = pipeline.run(data)

    # With the same three-tier caching as the framework:
    preprocessor = PreProcessor("per_unit", datasource, pipeline)
    data = preprocessor.pipeline(
        data_cache_path=".cache",
        data_library_part_path="./picid/data/datasources",
        transform_library_part_path="./picid/transforms",
        cache_preprocessed=True,
    )

Both ConfigTransformManager and TransformPipeline satisfy TransformSequenceProtocol.
PreProcessor and the pipeline runners depend only on that protocol.

Pipeline runners
----------------
    DirectPipelineRunner     — no caching; load → split → fetch → transform → return.
    CachingPipelineRunner    — three-tier cache (loaded_and_splitted_data → boundary →
                               preprocessed) with optional boundary restore.

Behaviour is identical to the previous implementation; the logic is decomposed
into named methods for testability and the concrete transform manager type is
no longer hardcoded.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union

from numpy import ndarray

from picid.data.cache.offline import FileSystemCache
from picid.data.data_objects.data import DatasetContainer
from picid.data.data_objects.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestEntry,
    MetadataManifest,
)
from picid.data.datasources.base.multi_source_loader import MultiSourceLoader
from picid.data.datasources.base.single_source_loader import SingleSourceLoader
from picid.data.preprocessing.base import PreProcessorInterface
from picid.exceptions import TransformError
from picid.transforms.base.data_transform import DataTransform
from picid.transforms.base.pipeline import _get_producer_version
from picid.transforms.base.transform_manager import ConfigTransformManager
from picid.transforms.base.transform_pipeline import (
    TransformPipeline,
    TransformSequenceProtocol,
)
from picid.utils import print_data_dict_structure
from picid.data.preprocessing.pipeline_runners import (
    CachingPipelineRunner,
    DirectPipelineRunner,
)
from picid.utils.rich_output import (
    descriptive_dict_differences_str,
    print_transforms_summary,
    to_descriptive_dict,
)

logger = logging.getLogger(__name__)

# Union type accepted wherever a transform sequence is expected.
AnyTransformSequence = Union[ConfigTransformManager, TransformPipeline]


# ---------------------------------------------------------------------------
# PreProcessor
# ---------------------------------------------------------------------------

class PreProcessor(PreProcessorInterface):
    """
    Orchestrates loading, splitting, transformation, and caching of datasets.

    Accepts either a ConfigTransformManager (framework path, YAML-driven) or a
    TransformPipeline (library path, Python-only) as the transforms argument.
    Both satisfy TransformSequenceProtocol and are handled identically by the
    pipeline runners.

    Attributes
    ----------
    mode : str
        "per_unit" or "cross_unit".
    datasource : SingleSourceLoader | MultiSourceLoader
    transforms : ConfigTransformManager | TransformPipeline
    data : Optional[DatasetContainer]
        None until pipeline() or fetch_data() + apply_transforms() is called.
    """

    def __init__(
        self,
        mode: str,
        datasource: Union[SingleSourceLoader, MultiSourceLoader],
        transforms: Optional[AnyTransformSequence] = None,
        **kwargs,
    ):
        if mode not in ("per_unit", "cross_unit"):
            raise ValueError(
                f"Unsupported mode: {mode!r}. "
                "Supported modes are 'per_unit' and 'cross_unit'."
            )

        if transforms is None:
            transforms = ConfigTransformManager(transforms_config=None)
            logger.info("No transforms provided; using empty ConfigTransformManager.")
        elif not isinstance(transforms, TransformSequenceProtocol):
            raise TypeError(
                f"transforms must be a ConfigTransformManager or TransformPipeline "
                f"(or any object satisfying TransformSequenceProtocol), "
                f"got {type(transforms).__name__}."
            )

        super().__init__(datasource=datasource, transforms=transforms)

        self.mode = mode
        self._is_preprocessed = False
        self.data: Optional[DatasetContainer] = None
        self.meta_data: dict = {}
        self.cache = FileSystemCache()

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def get_meta_data_dict(self) -> dict:
        return self.meta_data

    def get_processed_data_dict(
        self, return_splits_on_first_level: bool = False
    ) -> Union[DatasetContainer, Dict[str, Any]]:
        """
        Return the final processed data.

        When return_splits_on_first_level is True, returns:
            {split: {key: ndarray | list[ndarray]}}
        Otherwise returns the DatasetContainer directly.
        """
        if not self._is_preprocessed or self.data is None:
            raise RuntimeError(
                "Data has not been preprocessed. Run pipeline() first."
            )
        if return_splits_on_first_level:
            return self.data.to_split_dict()
        return self.data

    def get_cached_transform_manager(self) -> AnyTransformSequence:
        """Return the (possibly restored-from-cache) transform sequence."""
        return self.transforms

    # Backward-compatibility alias (typo preserved from original run.py usage).
    def get_chached_transform_manager(self) -> AnyTransformSequence:
        return self.get_cached_transform_manager()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def fetch_data(self) -> None:
        """Load data from the datasource and attach a manifest entry."""
        data = self.datasource.get_data()

        if not isinstance(data, DatasetContainer):
            raise TypeError(
                f"datasource.get_data() must return a DatasetContainer, "
                f"got {type(data).__name__}."
            )

        self.data = data
        self._add_datasource_manifest_entry()

    def _add_datasource_manifest_entry(self) -> None:
        """Append a datasource provenance entry to self.data.manifest if present."""
        if self.data is None:
            return
        manifest = getattr(self.data, "manifest", None)
        if not isinstance(manifest, MetadataManifest):
            return

        ds = self.datasource
        payload: Dict[str, Any] = {"datasource_type": type(ds).__name__}

        data_name = self._resolve_datasource_name(ds)
        if data_name is not None:
            payload["data_name"] = data_name
        if hasattr(ds, "task_mode"):
            payload["task_mode"] = getattr(ds, "task_mode", None)

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

    @staticmethod
    def _resolve_datasource_name(ds: Any) -> Optional[Any]:
        """
        Extract a data_name from a datasource using attribute/method precedence.
        Returns the name (str or list of str), or None if not found.
        """
        if hasattr(ds, "data_name"):
            val = getattr(ds, "data_name", None)
            if isinstance(val, str):
                return val

        if hasattr(ds, "get_data_name") and callable(getattr(ds, "get_data_name")):
            try:
                val = ds.get_data_name()
                if isinstance(val, str):
                    return val
                if isinstance(val, (list, tuple)) and all(
                    isinstance(x, str) for x in val
                ):
                    return val
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Transform application
    # ------------------------------------------------------------------

    def apply_transforms(
        self,
        data: DatasetContainer,
        transforms: Dict[str, DataTransform],
        after_each_transform_callback: Optional[
            Callable[[DatasetContainer, str], None]
        ] = None,
    ) -> DatasetContainer:
        """
        Apply an ordered dict of DataTransform objects to data sequentially.

        Parameters
        ----------
        data : DatasetContainer
        transforms : OrderedDict[str, DataTransform]
            Already-resolved DataTransform objects (from get_transforms()).
        after_each_transform_callback : callable, optional
            Called as callback(data, transform_name) after each successful
            transform.  Used internally for boundary cache saves.

        Returns
        -------
        DatasetContainer

        Raises
        ------
        ValueError
            If any value in transforms is not a DataTransform.
        TransformError
            If a transform raises during application.
        """
        if not all(isinstance(t, DataTransform) for t in transforms.values()):
            raise ValueError(
                "Every element of transforms must be a DataTransform instance."
            )

        summary = []

        try:
            for transform_name, transform in transforms.items():
                logger.info(
                    "Applying transform: '%s' (Strategy: %s)",
                    transform_name,
                    transform.strategy.__class__.__name__,
                )

                pre_descr = to_descriptive_dict(
                    data.to_split_dict()["train"], calculate_stat=False
                )
                start_time = time.time()

                try:
                    data, transform_log = transform.forward(data)
                except TransformError:
                    raise
                except Exception as e:
                    raise TransformError(
                        f"Transform {transform_name!r} failed. "
                        f"Original error: {e}",
                        step_id=transform_name,
                        cause=e,
                    ) from e

                if after_each_transform_callback is not None:
                    after_each_transform_callback(data, transform_name)

                elapsed = time.time() - start_time
                logger.info(
                    "Transform '%s' completed in %.4f seconds.",
                    transform.transform_name,
                    elapsed,
                )

                post_descr = to_descriptive_dict(
                    data.to_split_dict()["train"], calculate_stat=False
                )
                log_str = self._extract_log_string(transform_log)

                summary.append({
                    "transform_name": transform_name,
                    "time": f"{elapsed:.4f}",
                    "status": "Success",
                    "details": log_str,
                    "changes": str(descriptive_dict_differences_str(
                        pre_descr, post_descr, mode="changed"
                    )),
                    "added": str(descriptive_dict_differences_str(
                        pre_descr, post_descr, mode="added"
                    )),
                    "removed": str(descriptive_dict_differences_str(
                        pre_descr, post_descr, mode="removed"
                    )),
                    "inputs": str(pre_descr),
                })
        finally:
            print_transforms_summary(summary)

        return data

    @staticmethod
    def _extract_log_string(transform_log: Any) -> str:
        """
        Extract a human-readable string from a transform log dict.

        Prefers "train", falls back to "test", then the first available key.
        """
        if not transform_log:
            return "No additional transform log."
        if not isinstance(transform_log, dict):
            return str(transform_log)
        for preferred in ("train", "test"):
            if preferred in transform_log:
                return str(transform_log[preferred])
        first_key = next(iter(transform_log))
        logger.debug(
            "transform_log has no 'train' or 'test' key; using %r for summary.",
            first_key,
        )
        return str(transform_log[first_key])

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def pipeline(
        self,
        data_cache_path: str = None,
        data_library_part_path: str = None,
        transform_library_part_path: str = None,
        cache_preprocessed: bool = False,
        save_boundary_caches: bool = True,
    ) -> DatasetContainer:
        """
        Run the full preprocessing pipeline.

        When all cache paths and cache_preprocessed=True are provided, a
        CachingPipelineRunner is used.  Otherwise a DirectPipelineRunner is used.
        Both runners work with ConfigTransformManager and TransformPipeline.

        Parameters
        ----------
        data_cache_path : str, optional
            Root directory for all cache files.
        data_library_part_path : str, optional
            Path to datasource library (used for cache key computation).
        transform_library_part_path : str, optional
            Path to transform library (used for cache key computation).
        cache_preprocessed : bool
            Enable the preprocessed (final) cache.
        save_boundary_caches : bool
            When True (default), save a boundary cache after each cache-point
            transform so future runs can restore from there.

        Returns
        -------
        DatasetContainer
        """
        cache_enabled = bool(
            data_cache_path
            and transform_library_part_path
            and data_library_part_path
            and cache_preprocessed
        )

        if cache_enabled:
            runner: Union[DirectPipelineRunner, CachingPipelineRunner] = (
                CachingPipelineRunner(
                    preprocessor=self,
                    data_cache_path=data_cache_path,
                    data_library_part_path=data_library_part_path,
                    transform_library_part_path=transform_library_part_path,
                    save_boundary_caches=save_boundary_caches,
                )
            )
        else:
            runner = DirectPipelineRunner(self)

        self.data = runner.run()
        self._is_preprocessed = True

        logger.debug(
            "Data structure after transforms:\n%s",
            print_data_dict_structure(self.data),
        )
        return self.data
